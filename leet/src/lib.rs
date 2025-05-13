
use std::collections::HashMap;

pub fn is_unique_1_1_hashmap(string : String) -> bool {
    let mut map = HashMap::new();
    for c in string.chars() {
        if map.contains_key(&c) {
            return false;
        }
        map.insert(c, true);
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
}
