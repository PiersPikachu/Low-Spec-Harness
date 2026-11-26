//! calculator - exact arithmetic for the model (parsed, never eval'd), with Python's semantics: big integers, true
//! division `/`, floor division `//`, Python modulo, `**`, and the math functions the Python plugin allowed.

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero, One};

pub const NAME: &str = "calculator";
pub const DESCRIPTION: &str = "Evaluate an arithmetic expression exactly, e.g. '17*23', '(2**10 - 1) / 3', 'sqrt(2) * pi'. Supports + - * / // % **, parentheses, and math functions (sqrt, sin, cos, tan, log, log10, exp, floor, ceil, factorial) and constants pi, e.";

#[derive(Clone, Debug, PartialEq)]
pub enum N {
    I(BigInt),
    F(f64),
}

impl N {
    fn f(&self) -> f64 {
        match self {
            N::I(i) => i.to_f64().unwrap_or(f64::INFINITY),
            N::F(f) => *f,
        }
    }
    pub fn repr(&self) -> String {
        match self {
            N::I(i) => i.to_string(),
            N::F(f) => crate::util::py_float(*f),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum T {
    Num(N),
    Name(String),
    Op(&'static str),
    L,
    R,
    Comma,
}

fn lex(s: &str) -> Result<Vec<T>, String> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch.is_ascii_digit() || (ch == '.' && i + 1 < c.len() && c[i + 1].is_ascii_digit()) {
            let st = i;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '_') {
                i += 1;
            }
            let mut fl = false;
            if i < c.len() && c[i] == '.' {
                fl = true;
                i += 1;
                while i < c.len() && (c[i].is_ascii_digit() || c[i] == '_') {
                    i += 1;
                }
            }
            if i < c.len() && (c[i] == 'e' || c[i] == 'E') {
                let save = i;
                i += 1;
                if i < c.len() && (c[i] == '+' || c[i] == '-') {
                    i += 1;
                }
                if i < c.len() && c[i].is_ascii_digit() {
                    fl = true;
                    while i < c.len() && c[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let t: String = c[st..i].iter().filter(|&&x| x != '_').collect();
            out.push(T::Num(if fl { N::F(t.parse().map_err(|_| format!("invalid number {t}"))?) } else { N::I(t.parse().map_err(|_| format!("invalid number {t}"))?) }));
            continue;
        }
        if ch.is_alphabetic() || ch == '_' {
            let st = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(T::Name(c[st..i].iter().collect()));
            continue;
        }
        let two: String = c[i..(i + 2).min(c.len())].iter().collect();
        if two == "**" || two == "//" {
            out.push(T::Op(if two == "**" { "**" } else { "//" }));
            i += 2;
            continue;
        }
        out.push(match ch {
            '+' => T::Op("+"),
            '-' => T::Op("-"),
            '*' => T::Op("*"),
            '/' => T::Op("/"),
            '%' => T::Op("%"),
            '(' => T::L,
            ')' => T::R,
            ',' => T::Comma,
            _ => return Err(format!("invalid syntax at '{ch}'")),
        });
        i += 1;
    }
    Ok(out)
}

struct P {
    t: Vec<T>,
    i: usize,
}

fn zero_div(msg: &str) -> String {
    msg.to_string()
}

fn floor_div_int(a: &BigInt, b: &BigInt) -> BigInt {
    let (q, r) = (a / b, a % b);
    if !r.is_zero() && (r.is_negative() != b.is_negative()) {
        q - 1
    } else {
        q
    }
}
fn py_mod_int(a: &BigInt, b: &BigInt) -> BigInt {
    let r = a % b;
    if !r.is_zero() && (r.is_negative() != b.is_negative()) {
        r + b
    } else {
        r
    }
}
fn py_mod_f(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
        r + b
    } else if r == 0.0 {
        0.0f64.copysign(b)
    } else {
        r
    }
}

fn bin(op: &str, a: N, b: N) -> Result<N, String> {
    use N::*;
    Ok(match (op, a, b) {
        ("+", I(x), I(y)) => I(x + y),
        ("-", I(x), I(y)) => I(x - y),
        ("*", I(x), I(y)) => I(x * y),
        ("//", I(x), I(y)) => {
            if y.is_zero() {
                return Err(zero_div("integer division or modulo by zero"));
            }
            I(floor_div_int(&x, &y))
        }
        ("%", I(x), I(y)) => {
            if y.is_zero() {
                return Err(zero_div("integer modulo by zero"));
            }
            I(py_mod_int(&x, &y))
        }
        ("/", I(x), I(y)) => {
            if y.is_zero() {
                return Err(zero_div("division by zero"));
            }
            F(x.to_f64().unwrap_or(f64::NAN) / y.to_f64().unwrap_or(f64::NAN))
        }
        ("**", I(x), I(y)) => {
            if y.abs() > BigInt::from(10000) {
                return Err("exponent too large".into());
            }
            let e = y.to_i64().unwrap();
            if e >= 0 {
                I(num_traits::pow(x, e as usize))
            } else {
                if x.is_zero() {
                    return Err("0.0 cannot be raised to a negative power".into());
                }
                F(x.to_f64().unwrap_or(f64::NAN).powi(e as i32))
            }
        }
        (op, a, b) => {
            let (x, y) = (a.f(), b.f());
            if op == "**" && y.abs() > 10000.0 {
                return Err("exponent too large".into());
            }
            F(match op {
                "+" => x + y,
                "-" => x - y,
                "*" => x * y,
                "/" => {
                    if y == 0.0 {
                        return Err("float division by zero".into());
                    }
                    x / y
                }
                "//" => {
                    if y == 0.0 {
                        return Err("float floor division by zero".into());
                    }
                    ((x - py_mod_f(x, y)) / y).round()
                }
                "%" => {
                    if y == 0.0 {
                        return Err("float modulo".into());
                    }
                    py_mod_f(x, y)
                }
                "**" => {
                    if x < 0.0 && y.fract() != 0.0 {
                        return Err("complex result (negative number to a fractional power)".into());
                    }
                    if x == 0.0 && y < 0.0 {
                        return Err("0.0 cannot be raised to a negative power".into());
                    }
                    let r = x.powf(y);
                    if r.is_infinite() && x.is_finite() && y.is_finite() {
                        return Err("(34, 'Numerical result out of range')".into());
                    }
                    r
                }
                _ => unreachable!(),
            })
        }
    })
}

fn check(f: f64) -> Result<N, String> {
    if f.is_nan() {
        Err("math domain error".into())
    } else {
        Ok(N::F(f))
    }
}

fn call(name: &str, a: Vec<N>) -> Result<N, String> {
    let one = |a: &Vec<N>| -> Result<N, String> {
        if a.len() != 1 {
            return Err(format!("{name}() takes exactly one argument ({} given)", a.len()));
        }
        Ok(a[0].clone())
    };
    let fl = |x: f64| -> Result<N, String> { if x.is_finite() { Ok(N::I(BigInt::from(x as i128))) } else { Err("cannot convert float infinity to integer".into()) } };
    match name {
        "sqrt" => {
            let x = one(&a)?.f();
            if x < 0.0 {
                return Err("math domain error".into());
            }
            check(x.sqrt())
        }
        "sin" => check(one(&a)?.f().sin()),
        "cos" => check(one(&a)?.f().cos()),
        "tan" => check(one(&a)?.f().tan()),
        "exp" => {
            let r = one(&a)?.f().exp();
            if r.is_infinite() {
                return Err("math range error".into());
            }
            Ok(N::F(r))
        }
        "log" => {
            if a.is_empty() || a.len() > 2 {
                return Err("log expected 1 or 2 arguments".into());
            }
            let x = a[0].f();
            if x <= 0.0 {
                return Err("math domain error".into());
            }
            if a.len() == 2 {
                let b = a[1].f();
                if b <= 0.0 || b == 1.0 {
                    return Err(if b == 1.0 { "float division by zero".into() } else { "math domain error".into() });
                }
                return Ok(N::F(x.ln() / b.ln()));
            }
            Ok(N::F(x.ln()))
        }
        "log10" => {
            let x = one(&a)?.f();
            if x <= 0.0 {
                return Err("math domain error".into());
            }
            Ok(N::F(x.log10()))
        }
        "floor" => match one(&a)? {
            N::I(i) => Ok(N::I(i)),
            N::F(f) => fl(f.floor()),
        },
        "ceil" => match one(&a)? {
            N::I(i) => Ok(N::I(i)),
            N::F(f) => fl(f.ceil()),
        },
        "factorial" => match one(&a)? {
            N::I(i) => {
                if i.is_negative() {
                    return Err("factorial() not defined for negative values".into());
                }
                if i > BigInt::from(5000) {
                    return Err("factorial argument too large".into());
                }
                let n = i.to_u64().unwrap();
                let mut r = BigInt::one();
                for k in 2..=n {
                    r *= k;
                }
                Ok(N::I(r))
            }
            N::F(_) => Err("'float' object cannot be interpreted as an integer".into()),
        },
        "fabs" => Ok(N::F(one(&a)?.f().abs())),
        "abs" => match one(&a)? {
            N::I(i) => Ok(N::I(i.abs())),
            N::F(f) => Ok(N::F(f.abs())),
        },
        "round" => {
            if a.is_empty() || a.len() > 2 {
                return Err("round expected 1 or 2 arguments".into());
            }
            let nd = if a.len() == 2 {
                match &a[1] {
                    N::I(i) => Some(i.to_i64().unwrap_or(0)),
                    N::F(_) => return Err("'float' object cannot be interpreted as an integer".into()),
                }
            } else {
                None
            };
            match (&a[0], nd) {
                (N::I(i), _) => Ok(N::I(i.clone())),
                (N::F(f), None) => fl(round_half_even(*f)),
                (N::F(f), Some(d)) => {
                    let p = 10f64.powi(d as i32);
                    // Python rounds the exact decimal value; formatting with the requested digits does that
                    if d >= 0 {
                        Ok(N::F(format!("{:.*}", d as usize, f).parse().unwrap_or(round_half_even(f * p) / p)))
                    } else {
                        Ok(N::F(round_half_even(f * p) / p))
                    }
                }
            }
        }
        _ => Err(format!("not allowed: {name}")),
    }
}

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

impl P {
    fn peek(&self) -> Option<&T> {
        self.t.get(self.i)
    }
    fn next(&mut self) -> Option<T> {
        let t = self.t.get(self.i).cloned();
        self.i += 1;
        t
    }
    fn expr(&mut self) -> Result<N, String> {
        let mut a = self.term()?;
        while let Some(T::Op(o)) = self.peek().cloned() {
            if o != "+" && o != "-" {
                break;
            }
            self.i += 1;
            let b = self.term()?;
            a = bin(o, a, b)?;
        }
        Ok(a)
    }
    fn term(&mut self) -> Result<N, String> {
        let mut a = self.factor()?;
        while let Some(T::Op(o)) = self.peek().cloned() {
            if !matches!(o, "*" | "/" | "//" | "%") {
                break;
            }
            self.i += 1;
            let b = self.factor()?;
            a = bin(o, a, b)?;
        }
        Ok(a)
    }
    fn factor(&mut self) -> Result<N, String> {
        match self.peek() {
            Some(T::Op("-")) => {
                self.i += 1;
                Ok(match self.factor()? {
                    N::I(i) => N::I(-i),
                    N::F(f) => N::F(-f),
                })
            }
            Some(T::Op("+")) => {
                self.i += 1;
                self.factor()
            }
            _ => self.power(),
        }
    }
    fn power(&mut self) -> Result<N, String> {
        let a = self.atom()?;
        if let Some(T::Op("**")) = self.peek() {
            self.i += 1;
            let b = self.factor()?;
            return bin("**", a, b);
        }
        Ok(a)
    }
    fn atom(&mut self) -> Result<N, String> {
        match self.next() {
            Some(T::Num(n)) => Ok(n),
            Some(T::L) => {
                let v = self.expr()?;
                match self.next() {
                    Some(T::R) => Ok(v),
                    _ => Err("invalid syntax: missing ')'".into()),
                }
            }
            Some(T::Name(n)) => {
                if let Some(T::L) = self.peek() {
                    self.i += 1;
                    let mut args = Vec::new();
                    if let Some(T::R) = self.peek() {
                        self.i += 1;
                    } else {
                        loop {
                            args.push(self.expr()?);
                            match self.next() {
                                Some(T::Comma) => continue,
                                Some(T::R) => break,
                                _ => return Err("invalid syntax in call".into()),
                            }
                        }
                    }
                    return call(&n, args);
                }
                match n.as_str() {
                    "pi" => Ok(N::F(std::f64::consts::PI)),
                    "e" => Ok(N::F(std::f64::consts::E)),
                    _ => Err(format!("not allowed: name '{n}'")),
                }
            }
            t => Err(format!("invalid syntax{}", t.map(|x| format!(" near {x:?}")).unwrap_or_default())),
        }
    }
}

pub fn eval(expr: &str) -> Result<N, String> {
    let mut p = P { t: lex(&expr.replace('^', "**"))?, i: 0 };
    let v = p.expr()?;
    if p.i != p.t.len() {
        return Err("invalid syntax".into());
    }
    Ok(v)
}

pub fn run(expr: &str) -> String {
    match eval(expr) {
        Ok(v) => v.repr(),
        Err(e) => format!("error: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::run;
    #[test]
    fn python_semantics() {
        for (e, want) in [
            ("17*23", "391"),
            ("(2**10 - 1) / 3", "341.0"),
            ("2**100", "1267650600228229401496703205376"),
            ("-2**2", "-4"),
            ("2**-1", "0.5"),
            ("7//-2", "-4"),
            ("-7 % 3", "2"),
            ("7.5 % -2", "-0.5"),
            ("sqrt(2) * pi", "4.442882938158366"),
            ("1/3", "0.3333333333333333"),
            ("factorial(20)", "2432902008176640000"),
            ("floor(2.5) + ceil(2.1)", "5"),
            ("round(2.5)", "2"),
            ("round(2.675, 2)", "2.67"),
            ("1e16", "1e+16"),
            ("0.1+0.2", "0.30000000000000004"),
            ("2^3", "8"),
            ("1/0", "error: division by zero"),
            ("log(8, 2)", "3.0"),
        ] {
            assert_eq!(run(e), want, "{e}");
        }
    }
}
