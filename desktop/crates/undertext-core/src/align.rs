//! Character alignment used by the ASR stabilizer.
//!
//! `matching_blocks` gives the same blocks as difflib's `get_matching_blocks()` with
//! `autojunk=False`.

use std::collections::HashMap;

use crate::text::normalize_word;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    pub a: usize,
    pub b: usize,
    pub size: usize,
}

fn find_longest_match(
    a: &[char],
    b: &[char],
    b2j: &HashMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> Match {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for i in alo..ahi {
        let mut new_j2len: HashMap<usize, usize> = HashMap::new();
        if let Some(js) = b2j.get(&a[i]) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = j.checked_sub(1).and_then(|p| j2len.get(&p)).copied().unwrap_or(0) + 1;
                new_j2len.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2len = new_j2len;
    }
    // no junk heuristics
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        bestsize += 1;
    }
    while besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize] {
        bestsize += 1;
    }
    Match { a: besti, b: bestj, size: bestsize }
}

/// Matching blocks, sorted, adjacent blocks merged, ending with the `(la, lb, 0)` sentinel.
pub fn matching_blocks(a: &[char], b: &[char]) -> Vec<Match> {
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, c) in b.iter().enumerate() {
        b2j.entry(*c).or_default().push(j);
    }
    let (la, lb) = (a.len(), b.len());
    let mut queue = vec![(0, la, 0, lb)];
    let mut blocks = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let m = find_longest_match(a, b, &b2j, alo, ahi, blo, bhi);
        if m.size > 0 {
            blocks.push(m);
            if alo < m.a && blo < m.b {
                queue.push((alo, m.a, blo, m.b));
            }
            if m.a + m.size < ahi && m.b + m.size < bhi {
                queue.push((m.a + m.size, ahi, m.b + m.size, bhi));
            }
        }
    }
    blocks.sort();
    let mut out = Vec::new();
    let (mut i1, mut j1, mut k1) = (0, 0, 0);
    for m in blocks {
        if i1 + k1 == m.a && j1 + k1 == m.b {
            k1 += m.size;
        } else {
            if k1 > 0 {
                out.push(Match { a: i1, b: j1, size: k1 });
            }
            (i1, j1, k1) = (m.a, m.b, m.size);
        }
    }
    if k1 > 0 {
        out.push(Match { a: i1, b: j1, size: k1 });
    }
    out.push(Match { a: la, b: lb, size: 0 });
    out
}

/// Normalized characters of `words` and the char offset at which each word starts.
fn chars_of(words: &[String]) -> (Vec<char>, Vec<usize>) {
    let mut starts = Vec::with_capacity(words.len());
    let mut out = Vec::new();
    for w in words {
        starts.push(out.len());
        out.extend(normalize_word(w).chars());
    }
    (out, starts)
}

/// Words of `hypothesis` that come after the text in `committed`.
///
/// Offline models re-decoding a growing buffer revise the *start* of the hypothesis too
/// ("Yeah." becomes "Well,", "anymore" becomes "any more"), so word positions shift.
/// The end of the committed text is located by character-level alignment instead.
pub fn align_tail(committed: &[String], hypothesis: &[String]) -> Vec<String> {
    if committed.is_empty() {
        return hypothesis.to_vec();
    }
    let (ca, _) = chars_of(committed);
    let (hb, starts) = chars_of(hypothesis);
    if ca.is_empty() {
        return hypothesis.to_vec();
    }
    if hb.is_empty() {
        return Vec::new();
    }
    let blocks: Vec<Match> = matching_blocks(&ca, &hb).into_iter().filter(|m| m.size >= 2).collect();
    let Some(first) = blocks.first() else {
        // nothing recognisable: fall back to positional slicing
        return hypothesis.get(committed.len()..).unwrap_or(&[]).to_vec();
    };
    let mut last = *first;
    for m in &blocks[1..] {
        if m.a + m.size > last.a + last.size {
            last = *m;
        }
    }
    // unmatched committed characters after the last block still consume hypothesis text
    let end = (last.b + last.size + (ca.len() - (last.a + last.size))).min(hb.len());
    for (i, start) in starts.iter().enumerate() {
        if *start >= end {
            return hypothesis[i..].to_vec();
        }
    }
    Vec::new()
}
