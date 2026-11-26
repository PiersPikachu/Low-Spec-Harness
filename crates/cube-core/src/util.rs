//! Small helpers shared by the core.

/// Python `repr(float)`.
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // shortest round-trip digits, then Python's layout: exponent form when exp < -4 or exp >= 16
    let e = format!("{f:e}"); // e.g. "1.2345e-5", "1e16"
    let (mant, exp) = e.split_once('e').unwrap();
    let x: i32 = exp.parse().unwrap();
    if x < -4 || x >= 16 {
        let m = if mant.contains('.') { mant.to_string() } else { mant.to_string() };
        format!("{m}e{}{:02}", if x < 0 { '-' } else { '+' }, x.abs())
    } else {
        let neg = mant.starts_with('-');
        let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
        let mut s = String::new();
        if x >= 0 {
            let xi = x as usize;
            if digits.len() <= xi + 1 {
                s.push_str(&digits);
                for _ in digits.len()..=xi {
                    s.push('0');
                }
                s.push_str(".0");
            } else {
                s.push_str(&digits[..xi + 1]);
                s.push('.');
                s.push_str(&digits[xi + 1..]);
            }
        } else {
            s.push_str("0.");
            for _ in 0..(-x - 1) {
                s.push('0');
            }
            s.push_str(&digits);
        }
        if neg {
            format!("-{s}")
        } else {
            s
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn floats() {
        for (f, s) in [(1.0, "1.0"), (0.1, "0.1"), (1e16, "1e+16"), (1e15, "1000000000000000.0"), (1.5e-5, "1.5e-05"), (0.0001, "0.0001"), (-2.5, "-2.5"), (123.456, "123.456"), (1e-7, "1e-07"), (4.442882938158366, "4.442882938158366")] {
            assert_eq!(super::py_float(f), s);
        }
    }
}
