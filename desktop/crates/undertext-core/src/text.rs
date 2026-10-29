//! Text helpers shared by the ASR stabilizer, sentence tracker and translators.

/// Abbreviations that end with a period but usually do not end a sentence.
pub const ABBREVIATIONS: &[&str] = &[
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e.",
    "u.s.", "u.k.", "no.", "fig.", "approx.", "dept.", "inc.", "ltd.", "co.", "jan.", "feb.",
    "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.", "mt.",
    "a.m.", "p.m.",
];

/// Words that usually open a new clause; a good place to cut an over-long sentence.
pub const CLAUSE_OPENERS: &[&str] = &[
    "and", "but", "so", "because", "which", "that", "when", "while", "although", "though", "if",
    "or", "where", "who", "whereas", "since", "unless", "then",
];

const TRAILING_CLOSERS: &[char] = &['"', '\'', '”', '’', ')', ']', '}'];

/// Whitespace tokenization that keeps punctuation attached to words.
pub fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_owned).collect()
}

pub fn join_words<S: AsRef<str>>(words: &[S]) -> String {
    let mut out = String::new();
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(w.as_ref());
    }
    out
}

/// Case- and punctuation-insensitive form used to compare ASR hypotheses.
pub fn normalize_word(word: &str) -> String {
    word.chars()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '\'')
        .collect()
}

pub fn common_prefix_len<T, K: PartialEq>(a: &[T], b: &[T], key: impl Fn(&T) -> K) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| key(x) == key(y)).count()
}

fn strip_closers(word: &str) -> &str {
    word.trim_end_matches(TRAILING_CLOSERS)
}

/// True if `word` (with attached punctuation) ends an English sentence.
pub fn is_sentence_end_word(word: &str) -> bool {
    let core = strip_closers(word);
    let Some(last) = core.chars().last() else {
        return false;
    };
    if last == '?' || last == '!' {
        return true;
    }
    if core.ends_with("...") {
        return true;
    }
    if last != '.' {
        return false;
    }
    if ABBREVIATIONS.contains(&core.to_lowercase().as_str()) {
        return false;
    }
    // single-letter initials such as "J." in "J. Smith"
    let mut chars = core.chars();
    if core.chars().count() == 2 {
        let first = chars.next().unwrap();
        if first.is_alphabetic() && first.is_uppercase() {
            return false;
        }
    }
    true
}

/// True if a clause boundary follows `word` (comma, semicolon, colon, dash or sentence end).
pub fn is_clause_end_word(word: &str) -> bool {
    let core = strip_closers(word);
    let Some(last) = core.chars().last() else {
        return false;
    };
    if matches!(last, ',' | ';' | ':') || core.ends_with("--") || last == '—' {
        return true;
    }
    is_sentence_end_word(word)
}

/// Pick where to cut an over-long unpunctuated sentence: a count `k`
/// (1 <= k <= max_words) so that `words[..k]` becomes its own sentence.
pub fn best_split_point<S: AsRef<str>>(words: &[S], max_words: usize) -> usize {
    if words.len() <= max_words {
        return words.len();
    }
    let window = &words[..max_words];
    let min_k = std::cmp::max(1, max_words / 3); // avoid tiny fragments
    for k in (min_k..=window.len()).rev() {
        if is_clause_end_word(window[k - 1].as_ref()) {
            return k;
        }
    }
    for k in (min_k..window.len()).rev() {
        if CLAUSE_OPENERS.contains(&normalize_word(window[k].as_ref()).as_str()) {
            return k;
        }
    }
    max_words
}

pub fn is_cjk(ch: char) -> bool {
    let c = ch as u32;
    (0x4E00..=0x9FFF).contains(&c)
        || (0x3400..=0x4DBF).contains(&c)
        || (0x20000..=0x2A6DF).contains(&c)
        || (0xF900..=0xFAFF).contains(&c)
}
