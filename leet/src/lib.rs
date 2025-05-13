
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

pub fn urlify_1_3(string: &mut Vec<char>) {
    let original_length = string.len();
    let space_count = string.iter().filter(|&&c| c == ' ').count();
    let new_length = string.len() + space_count * 2;
    string.resize(new_length, char::default());

    let mut index = new_length - 1;
    for i in (0..original_length).rev() {
        if string[i] == ' ' {
            string[index] = '0';
            string[index - 1] = '2';
            string[index - 2] = '%';
            if index > 2 {
                index -= 3;
            }
        } else {
            string[index] = string[i];
            if index > 0 {
                index -= 1;
            }
        }
    }
}

// Problem 1.4: Palindrome Permutation
// Given a string, write a function to check if it is a permutation of a palindrome.

pub fn is_palindrome_permutation_1_4(string: String) -> bool {
    let mut chars_count_map: HashMap<char, usize> = HashMap::new();
    for c in string.chars() {
        if !c.is_ascii() {
            panic!("Only ASCII characters are allowed");
        }

        if c == ' ' {
            continue;
        }

        let count = chars_count_map.entry(c.to_ascii_lowercase()).or_insert(0);
        *count += 1;
    }
    let odd_count = chars_count_map.values().filter(|&&count| count % 2 != 0).count();
    odd_count <= 1
}

// Problem 1.5: One Away
// There are three types of edits that can be performed on strings: insert a character, remove a character, or replace a character. Given two strings, write a function to check if they are one edit (or zero edits) away.

pub fn one_away_1_5(string_a: String, string_b: String) -> bool {
    let (shorter, longer) = if string_a.len() < string_b.len() {
        (string_a, string_b)
    } else {
        (string_b, string_a)
    };

    let len_shorter = shorter.len();
    let len_longer = longer.len();

    // If the lengths are equal, check for a replacement
    if len_shorter == len_longer {
        let mut found_difference = false;
        for i in 0..len_shorter {
            if shorter.chars().nth(i) != longer.chars().nth(i) {
                if found_difference {
                    return false;
                }
                found_difference = true;
            }
        }
        return true;
    }

    // If the lengths differ by more than 1, return false
    if len_longer - len_shorter > 1 {
        return false;
    }

    // Check for a removal or insertion
    let mut found_difference = false;
    let mut index_longer = 0;
    for index_shorter in 0..shorter.len() {
        if shorter.chars().nth(index_shorter) != longer.chars().nth(index_longer) {
            if found_difference {
                return false;
            }

            found_difference = true;
            // Skip the index of the longer string
            index_longer += 1;
        }

        // Always increment the index of the longer string along with the shorter string
        index_longer += 1;
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
        let mut test_string = "Mr John Doe".chars().collect::<Vec<_>>();
        urlify_1_3(&mut test_string);
        let expected_result = "Mr%20John%20Doe".chars().collect::<Vec<_>>();
        assert_eq!(test_string, expected_result);
    }

    #[test]
    fn test_is_palindrome_permutation_1_4() {
        let test_string = String::from("Tact Coa");
        let result = is_palindrome_permutation_1_4(test_string);
        assert_eq!(result, true);
        let test_string = String::from("abcde");
        let result = is_palindrome_permutation_1_4(test_string);
        assert_eq!(result, false);
    }

    #[test]
    fn test_one_away_1_5() {
        let test_string_a = String::from("pale");
        let test_string_b = String::from("ple");
        let result = one_away_1_5(test_string_a, test_string_b);
        assert_eq!(result, true);
        let test_string_a = String::from("pales");
        let test_string_b = String::from("pale");
        let result = one_away_1_5(test_string_a, test_string_b);
        assert_eq!(result, true);
        let test_string_a = String::from("pale");
        let test_string_b = String::from("bale");
        let result = one_away_1_5(test_string_a, test_string_b);
        assert_eq!(result, true);
        let test_string_a = String::from("pale");
        let test_string_b = String::from("bake");
        let result = one_away_1_5(test_string_a, test_string_b);
        assert_eq!(result, false);
    }
}
