
use std::collections::HashMap;

pub fn is_unique_1_1_hashmap(string: String) -> bool {
    let mut map = HashMap::new();
    for c in string.chars() {
        if map.contains_key(&c) {
            return false;
        }
        map.insert(c, true);
    }
    true
}

pub fn is_unique_1_1_bit_vector(string: String) -> bool {
    let mut bit_vector: usize = 0;
    for c in string.chars() {
        let index = c as usize - 'A' as usize;
        if bit_vector & (1 << index) != 0 {
            return false;
        }
        bit_vector |= 1 << index;
    }
    true
}

pub fn is_unique_1_1_no_external_data_structure(string: String) -> bool {
    let mut chars: Vec<_> = string.chars().collect();
    chars.sort_unstable();
    for i in 0..chars.len() - 1 {
        if chars[i] == chars[i + 1] {
            return false;
        }
    }
    true
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_unique_1_1_hashmap() {
        let test_string = String::from("abcdefg");
        let result = is_unique_1_1_hashmap(test_string);
        assert_eq!(result, true);
        let test_string = String::from("abcdeafg");
        let result = is_unique_1_1_hashmap(test_string);
        assert_eq!(result, false);
    }

    #[test]
    fn test_is_unique_1_1_bit_vector() {
        let test_string = String::from("abcdefg");
        let result = is_unique_1_1_bit_vector(test_string);
        assert_eq!(result, true);
        let test_string = String::from("abcdeafg");
        let result = is_unique_1_1_bit_vector(test_string);
        assert_eq!(result, false);
    }

    #[test]
    fn test_is_unique_1_1_no_external_data_structure() {
        let test_string = String::from("abcdefg");
        let result = is_unique_1_1_no_external_data_structure(test_string);
        assert_eq!(result, true);
        let test_string = String::from("abcdeafg");
        let result = is_unique_1_1_no_external_data_structure(test_string);
        assert_eq!(result, false);
    }
}
