use std::fmt::Display;

use jlabel::Label;

use super::tree::Tree;

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    trees: Vec<Tree>,
    pdf: Vec<Vec<ModelParameter>>,
}

impl Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "\n{}",
            self.trees.iter().fold(String::new(), |acc, curr| {
                if acc.is_empty() {
                    format!(
                        "    #{}: {} -> {}",
                        curr.state,
                        curr.nodes.len(),
                        self.pdf[curr.state - 2].len()
                    )
                } else {
                    format!(
                        "{}\n    #{}: {} -> {}",
                        acc,
                        curr.state,
                        curr.nodes.len(),
                        self.pdf[curr.state - 2].len()
                    )
                }
            }),
        )?;
        Ok(())
    }
}

impl Model {
    pub fn new(trees: Vec<Tree>, pdf: Vec<Vec<ModelParameter>>) -> Self {
        Self { trees, pdf }
    }

    /// Get index of tree and PDF
    /// Returns (tree_index, pdf_index)
    pub fn get_index(&self, state_index: usize, label: &Label) -> (Option<usize>, Option<usize>) {
        let tree_index = self.find_tree_index(state_index);

        let tree = match tree_index {
            Some(idx) => &self.trees[idx],
            None => &self.trees[0],
        };

        let pdf_index = tree.search_node(label);

        (
            tree_index
                // Somehow hts_engine_API requires 2 to be added to tree index
                .map(|index| index + 2),
            pdf_index,
        )
    }
    fn find_tree_index(&self, state_index: usize) -> Option<usize> {
        self.trees
            .iter()
            .enumerate()
            .position(|(_, tree)| tree.state == state_index)
    }

    /// Get parameter using interpolation weight
    pub fn get_parameter(&self, state_index: usize, label: &Label) -> &ModelParameter {
        let (Some(tree_index), Some(pdf_index)) = self.get_index(state_index, label) else {
            todo!("index not found!")
        };

        &self.pdf[tree_index - 2][pdf_index - 1]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelParameter {
    // (mean, vari)
    pub parameters: Vec<(f64, f64)>,
    pub msd: Option<f64>,
}

impl ModelParameter {
    pub fn new(size: usize, is_msd: bool) -> Self {
        Self {
            parameters: vec![(0.0, 0.0); size],
            msd: if is_msd { Some(0.0) } else { None },
        }
    }

    pub fn from_linear(lin: Vec<f64>) -> Self {
        let len = lin.len() / 2;
        let mut parameters = Vec::with_capacity(len);
        for i in 0..len {
            parameters.push((lin[i], lin[i + len]))
        }
        Self {
            parameters,
            msd: lin.get(len * 2).copied(),
        }
    }

    pub fn add_assign(&mut self, weight: f64, rhs: &Self) {
        for (i, p) in rhs.parameters.iter().enumerate() {
            self.parameters[i].0 += weight * p.0;
            self.parameters[i].1 += weight * p.1;
        }
        if let (Some(msd), Some(rhs)) = (self.msd.as_mut(), rhs.msd) {
            *msd += weight * rhs;
        }
    }
}
