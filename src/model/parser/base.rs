use std::ops::{Range, RangeFrom, RangeTo};

use nom::{
    bytes::complete::{take_while, take_while1},
    error::{ErrorKind, ParseError},
    IResult,
};

const SEPARATOR_CHARS: &str = " \n";
const PATTERN_WILDCARD: &str = "*?";
const JPCOMMON_SYMBOLS: &str = "!#%&+-/:=@^_|";

pub trait ParseTarget
where
    Self: Sized
        + Clone
        + nom::Slice<Range<usize>>
        + nom::Slice<RangeFrom<usize>>
        + nom::Slice<RangeTo<usize>>
        + nom::InputIter
        + nom::InputLength
        + nom::InputTake
        + nom::InputTakeAtPosition
        + nom::Offset
        + nom::ParseTo<isize>
        + nom::Compare<&'static str>,
{
    fn parse_template<'a, F, E>(self, cond: F) -> IResult<Self, Self, E>
    where
        F: Fn(char) -> bool,
        E: ParseError<Self>;
    fn parse_template1<'a, F, E>(self, cond: F) -> IResult<Self, Self, E>
    where
        F: Fn(char) -> bool,
        E: ParseError<Self>;

    #[inline(always)]
    fn sp<'a, E: ParseError<Self>>(self) -> IResult<Self, Self, E> {
        Self::parse_template(self, |c| SEPARATOR_CHARS.contains(c))
    }
    #[inline(always)]
    fn sp1<'a, E: ParseError<Self>>(self) -> IResult<Self, Self, E> {
        Self::parse_template1(self, |c| SEPARATOR_CHARS.contains(c))
    }
    #[inline(always)]
    fn parse_identifier<E: ParseError<Self>>(self) -> IResult<Self, Self, E> {
        Self::parse_template1(self, |c: char| {
            c.is_ascii() && (c.is_alphanumeric() || c == '_')
        })
    }
    #[inline(always)]
    fn parse_pattern<E: ParseError<Self>>(self) -> IResult<Self, Self, E> {
        Self::parse_template1(self, |c: char| {
            c.is_ascii()
                && (c.is_alphanumeric()
                    || PATTERN_WILDCARD.contains(c)
                    || JPCOMMON_SYMBOLS.contains(c))
        })
    }
    #[inline(always)]
    fn parse_ascii<E: ParseError<Self>>(self) -> IResult<Self, Self, E> {
        Self::parse_template(self, |c: char| c.is_ascii() && c != '\n')
    }
    fn parse_ascii_to_string<E: ParseError<Self>>(&self) -> IResult<Self, String, E>;
}

impl ParseTarget for &str {
    #[inline(always)]
    fn parse_template<'a, F, E>(self, cond: F) -> IResult<Self, Self, E>
    where
        F: Fn(char) -> bool,
        E: ParseError<Self>,
    {
        take_while(cond)(self)
    }
    #[inline(always)]
    fn parse_template1<'a, F, E>(self, cond: F) -> IResult<Self, Self, E>
    where
        F: Fn(char) -> bool,
        E: ParseError<Self>,
    {
        take_while1(cond)(self)
    }

    fn parse_ascii_to_string<E: ParseError<Self>>(&self) -> IResult<Self, String, E> {
        Self::parse_ascii(self).map(|(rest, result)| (rest, result.to_string()))
    }
}

impl ParseTarget for &[u8] {
    #[inline(always)]
    fn parse_template<'a, F, E>(self, cond: F) -> IResult<Self, Self, E>
    where
        F: Fn(char) -> bool,
        E: ParseError<Self>,
    {
        take_while(|c| cond(c as char))(self)
    }
    #[inline(always)]
    fn parse_template1<'a, F, E>(self, cond: F) -> IResult<Self, Self, E>
    where
        F: Fn(char) -> bool,
        E: ParseError<Self>,
    {
        take_while1(|c| cond(c as char))(self)
    }

    fn parse_ascii_to_string<E: ParseError<Self>>(&self) -> IResult<Self, String, E> {
        Self::parse_ascii(self).and_then(|(rest, result)| {
            match String::from_utf8(result.to_vec()) {
                Ok(s) => Ok((rest, s)),
                Err(_) => Err(nom::Err::Failure(E::from_error_kind(
                    result,
                    ErrorKind::Char,
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use nom::error::VerboseError;

    use super::ParseTarget;

    #[test]
    fn ascii() {
        assert_eq!(
            "hogehoge\"\nfugafuga".parse_ascii::<VerboseError<&str>>(),
            Ok(("\nfugafuga", "hogehoge\""))
        );
    }

    #[test]
    fn pattern() {
        assert_eq!(
            "*=d/A:*".parse_pattern::<VerboseError<&str>>(),
            Ok(("", "*=d/A:*"))
        );
        assert_eq!(
            "*^i-*".parse_pattern::<VerboseError<&str>>(),
            Ok(("", "*^i-*"))
        );
    }
}
