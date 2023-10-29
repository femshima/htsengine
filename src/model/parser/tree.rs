use std::marker::PhantomData;

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::digit1,
    combinator::{cut, map, opt, recognize},
    error::{context, ContextError, ErrorKind, ParseError},
    multi::separated_list0,
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    AsChar, IResult,
};

use super::base::ParseTarget;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    pattern: Vec<String>,
    state: usize,
    nodes: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    id: isize,
    question_name: String,
    yes: TreeIndex,
    no: TreeIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeIndex {
    Node(isize),
    Pdf(isize),
}

#[derive(Debug, Clone)]
pub struct Question {
    name: String,
    patterns: Vec<String>,
}

pub struct TreeParser<T>(PhantomData<T>);

impl<S: ParseTarget> TreeParser<S>
where
    <S as nom::InputIter>::Item: nom::AsChar + Clone,
    <S as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
{
    pub fn parse_signed_digits<'a, E: ParseError<S> + ContextError<S>>(
        i: S,
    ) -> IResult<S, isize, E> {
        use nom::character::complete::char;
        recognize(pair(opt(char('-')), digit1))(i).and_then(|(rest, number)| {
            match number.parse_to() {
                Some(n) => Ok((rest, n)),
                None => Err(nom::Err::Error(E::from_error_kind(
                    number,
                    ErrorKind::Float,
                ))),
            }
        })
    }
    pub fn parse_question_ident<'a, E: ParseError<S> + ContextError<S>>(i: S) -> IResult<S, S, E> {
        i.parse_template1(|c| c.is_ascii() && !" \n".contains(c))
    }
    pub fn parse_question<'a, E: ParseError<S> + ContextError<S>>(
        i: S,
    ) -> IResult<S, (S, Vec<S>), E> {
        use nom::character::complete::char;
        preceded(
            terminated(tag("QS"), S::sp1),
            separated_pair(
                Self::parse_question_ident,
                S::sp1,
                context(
                    "pattern",
                    delimited(
                        pair(char('{'), S::sp),
                        separated_list0(
                            pair(char(','), S::sp),
                            cut(delimited(char('\"'), S::parse_pattern, char('\"'))),
                        ),
                        pair(S::sp, char('}')),
                    ),
                ),
            ),
        )(i)
    }
    pub fn parse_node<'a, E: ParseError<S> + ContextError<S>>(i: S) -> IResult<S, Node, E> {
        let pdf_index = move |i| {
            S::parse_identifier(i).and_then(|(rest, input)| {
                let mut id_str = String::new();
                for c in input.iter_elements() {
                    let c = c.as_char();
                    if c.is_ascii_digit() {
                        id_str.push(c)
                    } else {
                        id_str.clear();
                    }
                }
                match id_str.parse() {
                    Ok(i) => Ok((rest, TreeIndex::Pdf(i))),
                    Err(_) => Err(nom::Err::Error(E::from_error_kind(input, ErrorKind::Digit))),
                }
            })
        };
        let tree_index = move |i| {
            alt((
                map(Self::parse_signed_digits, |i| TreeIndex::Node(i)),
                pdf_index,
            ))(i)
        };
        let branch = move |i| {
            use nom::character::complete::char;
            cut(alt((
                tree_index,
                delimited(char('\"'), tree_index, char('\"')),
            )))(i)
        };

        tuple((
            preceded(S::sp, Self::parse_signed_digits),
            preceded(S::sp1, Self::parse_question_ident),
            preceded(S::sp1, branch),
            preceded(S::sp1, branch),
        ))(i)
        .and_then(|(rest, (id, question_name, yes, no))| {
            Ok((
                rest,
                Node {
                    id,
                    question_name: question_name.parse_ascii_to_string()?.1,
                    yes,
                    no,
                },
            ))
        })
    }
    pub fn parse_tree<'a, E: ParseError<S> + ContextError<S>>(i: S) -> IResult<S, Tree, E> {
        use nom::character::complete::char;
        tuple((
            preceded(
                S::sp,
                delimited(
                    char('{'),
                    separated_list0(char(','), |s| {
                        let (rest, s) = S::parse_pattern(s)?;
                        let (_, sstr) = S::parse_ascii_to_string(&s)?;
                        Ok((rest, sstr))
                    }),
                    char('}'),
                ),
            ),
            preceded(
                S::sp,
                delimited(char('['), Self::parse_signed_digits, char(']')),
            ),
            preceded(
                S::sp,
                delimited(
                    pair(char('{'), S::sp),
                    separated_list0(S::sp1, Self::parse_node),
                    pair(S::sp, char('}')),
                ),
            ),
        ))(i)
        .and_then(|(rest, (pattern, state, nodes))| {
            Ok((
                rest,
                Tree {
                    pattern,
                    state: state as usize,
                    nodes,
                },
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use nom::error::VerboseError;

    use super::{Node, Tree, TreeIndex, TreeParser};

    #[test]
    fn parse_question() {
        assert_eq!(
            TreeParser::parse_question::<VerboseError<&str>>(
                r#"QS C-Mora_diff_Acc-Type<=0 { "*/A:-??+*","*/A:-?+*","*/A:0+*" }"#
            ),
            Ok((
                "",
                (
                    "C-Mora_diff_Acc-Type<=0",
                    vec!["*/A:-??+*", "*/A:-?+*", "*/A:0+*"]
                )
            ))
        );
    }

    #[test]
    fn parse_node() {
        assert_eq!(
            TreeParser::parse_node::<VerboseError<&str>>(concat!(
                r#" -235 R-Phone_Boin_E                                       -236          "dur_s2_230" "#,
                "\n}",
            )),
            Ok((
                " \n}",
                Node {
                    id: -235,
                    question_name: "R-Phone_Boin_E".to_string(),
                    yes: TreeIndex::Node(-236),
                    no: TreeIndex::Pdf(230)
                }
            ))
        );
    }

    #[test]
    fn parse_tree() {
        assert_eq!(
            TreeParser::parse_tree::<VerboseError<&str>>(
                r#"{*}[2]
{
    0 Utt_Len_Mora<=28                                    "gv_lf0_1"          -1      
    -1 Utt_Len_Mora=18                                     "gv_lf0_3"       "gv_lf0_2" 
}"#
            ),
            Ok((
                "",
                Tree {
                    pattern: vec!["*".to_string()],
                    state: 2,
                    nodes: vec![
                        Node {
                            id: 0,
                            question_name: "Utt_Len_Mora<=28".to_string(),
                            yes: TreeIndex::Pdf(1),
                            no: TreeIndex::Node(-1)
                        },
                        Node {
                            id: -1,
                            question_name: "Utt_Len_Mora=18".to_string(),
                            yes: TreeIndex::Pdf(3),
                            no: TreeIndex::Pdf(2)
                        }
                    ]
                }
            ))
        );
    }
}
