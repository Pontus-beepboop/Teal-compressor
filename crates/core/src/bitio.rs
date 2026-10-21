//! MSB-first bit reader and writer used by the TICO RAW (JPEG XS) entropy coder.

use crate::{Error, Result};

#[inline]
fn load_be64(d: &[u8], at: usize) -> u64 {
    u64::from_be_bytes(d[at..at + 8].try_into().unwrap())
}

/// Cached reader. Reads past `end` return zeros; callers validate positions at
/// sub-packet boundaries with [`BitReader::skip_to`].
pub struct BitReader<'a> {
    data: &'a [u8],
    p: usize,
    end: usize,
    cache: u64,
    avail: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8], begin: usize, end: usize) -> Self {
        BitReader { data, p: begin, end, cache: 0, avail: 0 }
    }

    #[inline]
    fn refill(&mut self) {
        if self.p + 8 <= self.end {
            self.cache |= load_be64(self.data, self.p) >> self.avail;
            self.p += ((63 - self.avail) >> 3) as usize;
            self.avail |= 56;
        } else {
            while self.avail <= 56 {
                let b = if self.p < self.end { self.data[self.p] as u64 } else { 0 };
                if self.p < self.end + 8 {
                    self.p += 1;
                }
                self.cache |= b << (56 - self.avail);
                self.avail += 8;
            }
        }
    }

    /// Reads `n` bits, n <= 56.
    #[inline]
    pub fn read(&mut self, n: u32) -> u64 {
        if n == 0 {
            return 0;
        }
        if self.avail < n {
            self.refill();
        }
        let v = self.cache >> (64 - n);
        self.cache <<= n;
        self.avail -= n;
        v
    }

    #[inline]
    pub fn bit(&mut self) -> u32 {
        if self.avail == 0 {
            self.refill();
        }
        let v = (self.cache >> 63) as u32;
        self.cache <<= 1;
        self.avail -= 1;
        v
    }

    /// Unary code: count of ones before a zero (at most 31).
    #[inline]
    pub fn unary(&mut self) -> Result<u32> {
        if self.avail < 33 {
            self.refill();
        }
        let ones = (!self.cache).leading_zeros();
        if ones >= 32 {
            return Err(Error("bad unary code".into()));
        }
        self.cache <<= ones + 1;
        self.avail -= ones + 1;
        Ok(ones)
    }

    #[inline]
    pub fn bit_pos(&self) -> usize {
        self.p * 8 - self.avail as usize
    }

    #[inline]
    pub fn byte(&self) -> usize {
        (self.bit_pos() + 7) >> 3
    }

    #[inline]
    pub fn align(&mut self) {
        self.seek_bit((self.bit_pos() + 7) & !7);
    }

    pub fn skip_to(&mut self, byte: usize) -> Result<()> {
        if byte * 8 < self.bit_pos() || byte > self.end {
            return Err(Error("sub-packet overrun".into()));
        }
        self.seek_bit(byte * 8);
        Ok(())
    }

    fn seek_bit(&mut self, bit: usize) {
        self.p = bit >> 3;
        self.cache = 0;
        self.avail = 0;
        let skip = (bit & 7) as u32;
        if skip != 0 {
            self.refill();
            self.cache <<= skip;
            self.avail -= skip;
        }
    }
}

/// Appends bits to a byte vector; [`BitWriter::align`] flushes the last partial byte.
pub struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl<'a> BitWriter<'a> {
    pub fn new(out: &'a mut Vec<u8>) -> Self {
        BitWriter { out, acc: 0, nbits: 0 }
    }

    /// Writes the low `n` bits of `value`, n <= 32.
    #[inline]
    pub fn put(&mut self, value: u64, n: u32) {
        if n == 0 {
            return;
        }
        self.acc |= (value & ((1u64 << n) - 1)) << (64 - self.nbits - n);
        self.nbits += n;
        if self.nbits >= 32 {
            self.out.extend_from_slice(&((self.acc >> 32) as u32).to_be_bytes());
            self.acc <<= 32;
            self.nbits -= 32;
        }
    }

    /// n <= 64
    #[inline]
    pub fn put_long(&mut self, value: u64, n: u32) {
        if n > 32 {
            self.put(value >> 32, n - 32);
            self.put(value & 0xffff_ffff, 32);
        } else {
            self.put(value, n);
        }
    }

    #[inline]
    pub fn put_unary(&mut self, x: u32) {
        if x < 32 {
            self.put(((1u64 << x) - 1) << 1, x + 1);
        } else {
            for _ in 0..x {
                self.put(1, 1);
            }
            self.put(0, 1);
        }
    }

    pub fn align(&mut self) {
        if self.nbits & 7 != 0 {
            self.put(0, 8 - (self.nbits & 7));
        }
        while self.nbits >= 8 {
            self.out.push((self.acc >> 56) as u8);
            self.acc <<= 8;
            self.nbits -= 8;
        }
    }
}

impl Drop for BitWriter<'_> {
    fn drop(&mut self) {
        self.align();
    }
}

/// Bit interleaving for a code group's 4 magnitudes: bit i of m goes to bit 4i.
#[inline]
pub fn spread16(m: u32) -> u64 {
    let mut x = (m & 0xffff) as u64;
    x = (x | (x << 24)) & 0x0000_00ff_0000_00ff;
    x = (x | (x << 12)) & 0x000f_000f_000f_000f;
    x = (x | (x << 6)) & 0x0303_0303_0303_0303;
    x = (x | (x << 3)) & 0x1111_1111_1111_1111;
    x
}

#[inline]
pub fn compact16(mut x: u64) -> u32 {
    x &= 0x1111_1111_1111_1111;
    x = (x | (x >> 3)) & 0x0303_0303_0303_0303;
    x = (x | (x >> 6)) & 0x000f_000f_000f_000f;
    x = (x | (x >> 12)) & 0x0000_00ff_0000_00ff;
    x = (x | (x >> 24)) & 0xffff;
    x as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn bits_round_trip() {
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        let items: Vec<(u64, u32, bool)> = (0..20_000)
            .map(|_| {
                let r = rng(&mut seed);
                let n = (r % 33) as u32;
                let v = if n == 0 { 0 } else { (r >> 8) & ((1u64 << n) - 1) };
                (v, n, r & 1 == 0 && n < 32) // unary codes are at most 31
            })
            .collect();
        let mut out = Vec::new();
        {
            let mut w = BitWriter::new(&mut out);
            for &(v, n, unary) in &items {
                if unary {
                    w.put_unary(n);
                } else {
                    w.put(v, n);
                }
            }
            w.align();
        }
        let mut r = BitReader::new(&out, 0, out.len());
        for &(v, n, unary) in &items {
            if unary {
                assert_eq!(r.unary().unwrap(), n);
            } else {
                assert_eq!(r.read(n), v);
            }
        }
    }

    #[test]
    fn spread_compact_round_trip() {
        for m in [0u32, 1, 0x8000, 0xffff, 0x1234, 0xa5a5] {
            assert_eq!(compact16(spread16(m)), m);
        }
    }
}
