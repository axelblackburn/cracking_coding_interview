
use std::collections::HashMap;


// Problem 1.1: Is Unique
// Implement an algorithm to determine if a string has all unique characters.

// O(n)
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

// O(n)
pub fn is_unique_1_1_bit_vector(string: String) -> bool {
    let mut bit_vector: usize = 0;
    for c in string.chars() {
        if !c.is_ascii_alphabetic() {
            panic!("Only alphabetic characters are allowed");
        }

        let index = c as usize - 'A' as usize;
        if bit_vector & (1 << index) != 0 {
            return false;
        }
        bit_vector |= 1 << index;
    }
    true
}

// O(n log n)
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

// Problem 1.2: Check Permutation
// Given two strings, write a method to decide if one is a permutation of the other.

// O(n log n)
pub fn is_permutation_1_2(string_a: String, string_b: String) -> bool {
    if string_a.len() != string_b.len() {
        return false;
    }

    let mut chars_a: Vec<_> = string_a.chars().collect();
    chars_a.sort_unstable();
    let mut chars_b: Vec<_> = string_b.chars().collect();
    chars_b.sort_unstable();
    for i in 0..chars_a.len() {
        if chars_a[i] != chars_b[i] {
            return false;
        }
    }
    true
}

// O(n)
pub fn is_permutation_1_2_hashmap(string_a: String, string_b: String) -> bool {
    if string_a.len() != string_b.len() {
        return false;
    }

    let mut map: HashMap<char, usize> = HashMap::new();
    for c in string_a.chars() {
        if map.contains_key(&c) {
            *map.get_mut(&c).unwrap() += 1;
        } else {
            map.insert(c, 1);
        }
    }

    for c in string_b.chars() {
        if let Some(count) = map.get_mut(&c) {
            *count -= 1;
            if *count == 0 {
                map.remove(&c);
            }
        } else {
            return false;
        }
    }
    true
}

// Problem 1.3: URLify
// Write a method to replace all spaces in a string with '%20'. You may assume that the string has sufficient space at the end of the string to hold the additional characters, and that you are given the "true" length of the string.

pub fn urlify(string: String, true_length: usize) -> String {
    let mut result = String::new();
    for i in 0..true_length {
        let c = string.chars().nth(i).unwrap();
        if c == ' ' {
            result.push_str("%20");
        } else {
            result.push(c);
        }
    }
    result
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

    #[test]
    fn test_is_permutation_1_2() {
        let test_string_a = String::from("abcde");
        let test_string_b = String::from("edcba");
        let result = is_permutation_1_2(test_string_a, test_string_b);
        assert_eq!(result, true);
        let test_string_a = String::from("abcde");
        let test_string_b = String::from("edcbaa");
        let result = is_permutation_1_2(test_string_a, test_string_b);
        assert_eq!(result, false);
        let test_string_a = String::from("abcde");
        let test_string_b = String::from("edcbe");
        let result = is_permutation_1_2(test_string_a, test_string_b);
        assert_eq!(result, false);
    }

    #[test]
    fn test_is_permutation_1_2_hashmap() {
        let test_string_a = String::from("abcde");
        let test_string_b = String::from("edcba");
        let result = is_permutation_1_2_hashmap(test_string_a, test_string_b);
        assert_eq!(result, true);
        let test_string_a = String::from("abcde");
        let test_string_b = String::from("edcbaa");
        let result = is_permutation_1_2_hashmap(test_string_a, test_string_b);
        assert_eq!(result, false);
        let test_string_a = String::from("abcde");
        let test_string_b = String::from("edcbe");
        let result = is_permutation_1_2_hashmap(test_string_a, test_string_b);
        assert_eq!(result, false);
    }

    #[test]
    fn test_urlify() {
        let test_string = String::from("Mr John Smith    ");
        let true_length = 13;
        let result = urlify(test_string, true_length);
        assert_eq!(result, "Mr%20John%20Smith");
    }
}
