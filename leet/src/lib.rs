
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::LinkedList;
use std::rc::Rc;
use std::cell::RefCell;


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

// Problem 1.6: String Compression
// Implement a method to perform basic string compression using the counts of repeated characters. For example, the string "aabcccccaaa" would become "a2b1c5a3". If the "compressed" string would not become smaller than the original string, your method should return the original string.
// You can assume the string has only uppercase and lowercase letters (a-z).

pub fn string_compression_1_6(string: String) -> String {
    let mut compressed_string = String::new();
    let mut count = 1;

    for i in 0..string.len() {
        if i + 1 < string.len() && string.chars().nth(i) == string.chars().nth(i + 1) {
            count += 1;
        } else {
            compressed_string.push(string.chars().nth(i).unwrap());
            compressed_string.push_str(&count.to_string());
            count = 1;
        }
    }

    if compressed_string.len() < string.len() {
        compressed_string
    } else {
        string
    }
}

// Problem 1.7: Rotate Matrix
// Given an image represented by an NxN matrix, where each pixel in the image is 4 bytes, write a method to rotate the image by 90 degrees. Can you do this in place?

pub fn rotate_matrix_1_7(matrix: &mut Vec<Vec<i32>>) {
    let n = matrix.len();
    for layer in 0..n / 2 {
        let first = layer;
        let last = n - layer - 1;
        for i in first..last {
            let offset = i - first;
            let top_left = matrix[first][i];

            // bottom left -> top left
            matrix[first][i] = matrix[last - offset][first];

            // bottom right -> bottom left
            matrix[last - offset][first] = matrix[last][last - offset];

            // top right -> bottom right
            matrix[last][last - offset] = matrix[i][last];

            // top left -> top right
            matrix[i][last] = top_left;
        }
    }
}

// Problem 1.8: Zero Matrix
// Write an algorithm such that if an element in an MxN matrix is 0, its entire row and column are set to 0.

pub fn zero_matrix_1_8(matrix: &mut Vec<Vec<i32>>) {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut zero_rows = vec![false; rows];
    let mut zero_cols = vec![false; cols];

    // First pass: find all zeros
    for i in 0..rows {
        for j in 0..cols {
            if matrix[i][j] == 0 {
                zero_rows[i] = true;
                zero_cols[j] = true;
            }
        }
    }

    // Second pass: set rows and columns to zero
    for i in 0..rows {
        for j in 0..cols {
            if zero_rows[i] || zero_cols[j] {
                matrix[i][j] = 0;
            }
        }
    }
}

// Problem 1.9: String Rotation
// Assume you have a method isSubstring which checks if one word is a substring of another. Given two strings, s1 and s2, write code to check if s2 is a rotation of s1 using only one call to isSubstring (i.e., "waterbottle" is a rotation of "erbottlewat").

pub fn is_rotation_1_9(s1: String, s2: String) -> bool {
    if s1.len() != s2.len() || s1.is_empty() {
        return false;
    }
    let s1s1 = format!("{}{}", s1, s1);

    s1s1.contains(&s2)
}

// Linked List Implementation

#[derive(Clone, Debug)]
pub struct Node<T> {
    value: T,
    next: Option<Box<Node<T>>>,
}

#[derive(Clone, Debug)]
pub struct SinglyLinkedList<T> {
    head: Option<Box<Node<T>>>,
}

impl<T> SinglyLinkedList<T> {
    pub fn new() -> Self {
        SinglyLinkedList { head: None }
    }

    pub fn append(&mut self, value: T) {
        let new_node = Box::new(Node {
            value,
            next: None,
        });

        if let Some(ref mut head) = self.head {
            let mut current = head;
            while let Some(ref mut next) = current.next {
                current = next;
            }
            current.next = Some(new_node);
        } else {
            self.head = Some(new_node);
        }
    }
}

impl<T: Clone> SinglyLinkedList<T> {
    pub fn to_vector(&self) -> Vec<T> {
        let mut result = Vec::new();
        let mut current = &self.head;
        while let Some(node) = current {
            result.push(node.value.clone());
            current = &node.next;
        }
        result
    }
}

impl<T> Default for SinglyLinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Problem 2.1: Remove Duplicates from unsorted linked list

pub fn remove_duplicate_2_1(list: &mut SinglyLinkedList<i32>) {
    let mut seen = HashSet::new();
    let mut current = &mut list.head;

    while current.is_some() {
        match current {
            Some(node) if seen.contains(&node.value) => {
                *current = node.next.take();
            }
            Some(node) => {
                seen.insert(node.value);
                current = &mut node.next;
            }
            None => break,
        }
    }
}


// Problem 2.2: Return Kth to Last
// Implement an algorithm to find the kth to last element of a singly linked list.

pub fn kth_to_last_2_2(list: &mut SinglyLinkedList<i32>, k_to_last: usize) -> Result<i32, String> {
    let mut current = &list.head;
    let mut k_ahead = &list.head;

    // Move k_ahead k elements ahead
    for _i in 0..k_to_last {
        match k_ahead {
            Some(node) => k_ahead = &node.next,
            None => return Err(format!("Less than {k_to_last} elements in the list!")),
        }
    }

    // Move both pointers until the k_ahead hits the end
    while let Some(node) = k_ahead {
        k_ahead = &node.next;
        current = &current.as_ref().unwrap().next;
    }

    current.as_ref()
           .map(|node| node.value)
           .ok_or_else(|| "List is empty".to_string())
}

// Problem 2.3: Delete Middle Node
// Implement an algorithm to delete a node in the middle (i.e., any node but the first and last node, not necessarily the exact middle) of a singly linked list, given only access to that node.

pub fn delete_middle_node_2_3(node: &mut Node<i32>) {
    if let Some(ref mut next_node) = node.next {
        node.value = next_node.value;
        node.next = next_node.next.take();
    }
}


// Problem 2.4: Partition
// Write code to partition a linked list around a value x, such that all nodes less than x come before all nodes greater than or equal to x. If x is contained within the list, the values of x only need to be after the elements less than x (see below). The partition element x can appear anywhere in the "right partition"; it does not need to appear between the left and right partitions.

pub fn partition_2_4(list: &mut SinglyLinkedList<i32>, x: i32) {
    let mut before_head: Option<Box<Node<i32>>> = None;
    let mut before_tail: Option<&mut Box<Node<i32>>> = None;

    let mut after_head: Option<Box<Node<i32>>> = None;
    let mut after_tail: Option<&mut Box<Node<i32>>> = None;

    let mut runner = list.head.take();

    while let Some(mut node) = runner {
        runner = node.next.take();

        if node.value < x {
            if let Some(tail) = before_tail {
                tail.next = Some(node);
                before_tail = tail.next.as_mut();
            } else {
                before_head = Some(node);
                before_tail = before_head.as_mut();
            }
        } else {
            if let Some(tail) = after_tail {
                tail.next = Some(node);
                after_tail = tail.next.as_mut();
            } else {
                after_head = Some(node);
                after_tail = after_head.as_mut();
            }
        }
    }

    if let Some(tail) = before_tail {
        tail.next = after_head;
        list.head = before_head;
    } else {
        list.head = after_head;
    }
}

// Problem 2.5: Sum Lists
// You have two numbers represented by a linked list, where each node contains a single digit. The digits are stored in reverse order, such that the 1's digit is at the head of the list. Write a function that adds the two numbers and returns it as a linked list.

pub fn sum_lists_1s_first_2_5(list_a: &SinglyLinkedList<u8>, list_b: &SinglyLinkedList<u8>) -> SinglyLinkedList<u8> {
    fn list_to_number(list: &SinglyLinkedList<u8>) -> u32 {
        let mut n: u32 = 0;
        let mut multiplier: u32 = 1;
        let mut head = &list.head;
        while let Some(node) = head {
            head = &node.next;
            n += node.value as u32 * multiplier;
            multiplier *= 10;
        }
        n
    }
    let sum = list_to_number(list_a) + list_to_number(list_b);

    fn number_to_list(n: u32) -> SinglyLinkedList<u8> {
        let mut list = SinglyLinkedList::new();
        let mut last: Option<&mut Box<Node<u8>>> = None;
        let mut n = n;
        while n > 0 {
            let digit = (n - (n / 10 * 10)) as u8;
            let new_node = Some(Box::new(Node {value: digit, next: None}));
            if let Some(last_node) = last {
                last_node.next = new_node;
                last = last_node.next.as_mut();
            } else {
                list.head = new_node;
                last = list.head.as_mut();
            }
            n /= 10;
        }
        list
    }
    number_to_list(sum)
}

pub fn sum_lists_1s_last_2_5(list_a: &SinglyLinkedList<u8>, list_b: &SinglyLinkedList<u8>) -> SinglyLinkedList<u8> {
    fn list_to_number(list: &SinglyLinkedList<u8>) -> u32 {
        let mut n: u32 = 0;
        let mut head = &list.head;
        while let Some(node) = head {
            head = &node.next;
            n *= 10;
            n += node.value as u32;
        }
        n
    }
    let sum = list_to_number(list_a) + list_to_number(list_b);

    fn number_to_list(n: u32) -> SinglyLinkedList<u8> {
        let mut list = SinglyLinkedList::new();
        let mut first: Option<Box<Node<u8>>> = None;
        let mut n = n;
        while n > 0 {
            let digit = (n - (n / 10 * 10)) as u8;
            let new_node = Some(Box::new(Node {value: digit, next: first}));
            first = new_node;
            n /= 10;
        }
        list.head = first;
        list
    }
    number_to_list(sum)
}

// Problem 2.6: Palindrome
// Implement a function to check if a linked list is a palindrome.

// Odd
// 1 2 3 4 5 6 7 8 9 N
// SF                  C=0
// _ S F               C=1
// _ _ S _ F           C=2
// _ _ _ S _ _ F       C=3
// _ _ _ _ S _ _ _ F   C=4
// _ _ _ _ M S _ _ _ F

// Even
// 1 2 3 4 5 6 7 8 N
// SF                  C=0
// _ S F               C=1
// _ _ S _ F           C=2
// _ _ _ S _ _ F       C=3
// _ _ _ _ S _ _ _ F   C=4

pub fn palindrome_2_6(list: SinglyLinkedList<i32>) -> bool {
    // Using a runner and a runner twice as fast, we can find the middle
    let mut slow_runner: &Option<Box<Node<i32>>> = &list.head;
    let mut fast_runner: &Option<Box<Node<i32>>> = slow_runner;
    let mut half_length: usize = 0;

    // Using a second list built backwards, we can test equality of the 2 sides
    let mut inverted_half: Option<Box<Node<i32>>> = None;

    while let Some(find_end_node) = fast_runner {
        // Advance the fast runner once
        fast_runner = &find_end_node.next;

        // Advance the fast runner twice
        if let Some(find_end_node) = &find_end_node.next {
            fast_runner = &find_end_node.next;
            // We advanced fast twice, we can increase the half length
            half_length += 1;
            // Advance the slow runner once
            if let Some(node) = slow_runner {
                // Prepend the current slow in the inverted list
                let new_node = Box::new(Node { value: node.value, next: inverted_half });
                inverted_half = Some(new_node);
                slow_runner = &node.next;
            } else {
                panic!("Something went really wrong");
            }
        } else {
            // We have an odd number of elements, so we need to skip the middle
            if let Some(node) = slow_runner {
                slow_runner = &node.next;
            } else {
                panic!("Something went really wrong");
            }
        }
    }

    let mut option_node_a: &Option<Box<Node<i32>>> = &inverted_half;
    let mut option_node_b: &Option<Box<Node<i32>>> = slow_runner;
    for _i in 0..half_length {
        if let Some(node_a) = option_node_a {
            if let Some(node_b) = option_node_b {
                if node_a.value != node_b.value {
                    return false;
                }
                option_node_a = &node_a.next;
                option_node_b = &node_b.next;
            } else {
                panic!("Something went wrong");
            }
        } else {
            panic!("Something went wrong");
        }
    }

    true
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RcNode<T> {
    value: T,
    next: Option<Rc<RefCell<RcNode<T>>>>,
}

#[derive(Clone, Debug)]
pub struct RcSinglyLinkedList<T> {
    head: Option<Rc<RefCell<RcNode<T>>>>,
}

impl<T> RcSinglyLinkedList<T> {
    pub fn new() -> Self {
        RcSinglyLinkedList { head: None }
    }

    pub fn append(&mut self, value: T) {
        let new_node = Rc::new(RefCell::new(RcNode {
            value,
            next: None,
        }));

        match self.head.as_ref() {
            None => {
                self.head = Some(new_node);
            }
            Some(head) => {
                let mut current = Rc::clone(head);
                loop {
                    let next = current.borrow().next.clone();
                    match next {
                        Some(next_node) => {
                            current = next_node;
                        }
                        None => {
                            current.borrow_mut().next = Some(new_node);
                            break;
                        }
                    }
                }
            }
        }
    }
}

// Problem 2.7: Intersection
// Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting node. Note that the intersection is defined based on reference, not value. That is, if the kth node of the first linked list is the exact same node (by reference) as the jth node of the second linked list, then they are intersecting.

pub fn intersection_2_7(list_a: &RcSinglyLinkedList<i32>, list_b: &RcSinglyLinkedList<i32>) -> Option<Rc<RefCell<RcNode<i32>>>> {
    fn get_length_and_tail(mut head: Option<Rc<RefCell<RcNode<i32>>>>) -> (usize, Option<Rc<RefCell<RcNode<i32>>>>) {
        let mut length = 0;
        let mut tail = None;
        while let Some(node) = head {
            length += 1;
            tail = Some(node.clone());
            head = node.borrow().next.clone();
        }
        (length, tail)
    }

    let (len_a, tail_a) = get_length_and_tail(list_a.head.clone());
    let (len_b, tail_b) = get_length_and_tail(list_b.head.clone());

    // If tails are not the same, there is no intersection
    if !Rc::ptr_eq(tail_a.as_ref().unwrap(), tail_b.as_ref().unwrap()) {
        return None;
    }

    let mut current_a = list_a.head.clone();
    let mut current_b = list_b.head.clone();

    // Align the starting points
    if len_a > len_b {
        for _ in 0..(len_a - len_b) {
            current_a = current_a.unwrap().borrow().next.clone();
        }
    } else {
        for _ in 0..(len_b - len_a) {
            current_b = current_b.unwrap().borrow().next.clone();
        }
    }

    // Compare nodes for intersection
    while let (Some(node_a), Some(node_b)) = (current_a.clone(), current_b.clone()) {
        if Rc::ptr_eq(&node_a, &node_b) {
            return Some(node_a);
        }
        current_a = node_a.borrow().next.clone();
        current_b = node_b.borrow().next.clone();
    }

    None
}


// Problem 2.8: Loop Detection
// Given a circular linked list, implement an algorithm that returns the node at the beginning of the loop.

pub fn loop_detection_2_8(
    list: &RcSinglyLinkedList<i32>,
) -> Option<Rc<RefCell<RcNode<i32>>>> {
    let mut slow = list.head.clone();
    let mut fast = list.head.clone();

    // Phase 1: Detect the loop
    loop {
        slow = slow.and_then(|node| node.borrow().next.clone());
        fast = fast
            .and_then(|node| node.borrow().next.clone())
            .and_then(|node| node.borrow().next.clone());

        match (&slow, &fast) {
            (Some(s), Some(f)) => {
                if Rc::ptr_eq(s, f) {
                    break;
                }
            }
            _ => return None, // No loop
        }
    }

    // Phase 2: Find the entry point of the loop
    let mut start = list.head.clone();
    while let (Some(s_node), Some(f_node)) = (&start, &slow) {
        if Rc::ptr_eq(s_node, f_node) {
            return Some(Rc::clone(s_node));
        }

        // Extract `next` values first to avoid simultaneous borrow and assignment
        let s_next = s_node.borrow().next.clone();
        let f_next = f_node.borrow().next.clone();
        start = s_next;
        slow = f_next;
    }

    None
}

// Problem 3.1 : Three in One
// Describe how you could use a single array to implement three stacks.

// Simple solution: 3 thirds

#[derive(Debug)]
pub struct StacksOnArray<T> {
    how_many_stacks: usize,
    stacks_size: usize,
    array: Vec<T>,
    stacks_counts: Vec<usize>,
}

impl<T: Copy + Default> StacksOnArray<T> {
    pub fn new(stacks_size: usize) -> Self {
        let mut stack = StacksOnArray {
            how_many_stacks: 3,
            stacks_size: stacks_size,
            array: Vec::<T>::new(),
            stacks_counts: Vec::new(),
        };
        stack.array.resize(stack.how_many_stacks * stack.stacks_size, T::default());
        stack.stacks_counts.resize(stack.how_many_stacks, 0);

        stack
    }

    pub fn is_full(&self, stack_id: usize) -> bool {
        if stack_id >= self.how_many_stacks {
            panic!("Invalid stack id {stack_id}");
        }

        self.stacks_counts[stack_id] >= self.stacks_size
    }

    pub fn is_empty(&self, stack_id: usize) -> bool {
        if stack_id >= self.how_many_stacks {
            panic!("Invalid stack id {stack_id}");
        }

        self.stacks_counts[stack_id] == 0
    }

    pub fn push(&mut self, stack_id: usize, value: T) {
        if stack_id >= self.how_many_stacks {
            panic!("Invalid stack id {stack_id}");
        }

        if self.is_full(stack_id) {
            panic!("Stack {stack_id} is full");
        }

        self.array[stack_id * self.stacks_size + self.stacks_counts[stack_id]] = value;
        self.stacks_counts[stack_id] += 1;
    }

    pub fn pop(&mut self, stack_id: usize) -> T {
        let value = self.peek(stack_id);
        self.stacks_counts[stack_id] -= 1;

        value
    }

    pub fn peek(&self, stack_id: usize) -> T {
        if stack_id >= self.how_many_stacks {
            panic!("Invalid stack id {stack_id}");
        }

        if self.is_empty(stack_id) {
            panic!("Stack {stack_id} is empty");
        }

        self.array[stack_id * self.stacks_size + self.stacks_counts[stack_id] - 1]
    }
}

// Problem 3.2: Stack Min
// How would you design a stack which, in addition to push and pop, has a function min which returns the minimum element? Push, pop and min should all operate in O(1) time.

#[derive(Debug)]
struct StackWMinNode {
    value: i32,
    min_below: i32,
}

#[derive(Debug)]
pub struct StackWMin {
    list: LinkedList<StackWMinNode>,
}

impl StackWMin {
    pub fn new() -> Self {
        StackWMin { list: LinkedList::new() }
    }

    pub fn push(&mut self, value: i32) {
        let previous_min = if self.list.is_empty() {
            value
        } else {
            self.list.front().unwrap().min_below
        };
        self.list.push_front(StackWMinNode { value: value, min_below: previous_min.min(value) });
    }

    pub fn pop(&mut self) -> i32 {
        match self.list.pop_front() {
            None => panic!("Cannot pop from empty list"),
            Some(node) => node.value
        }
    }

    pub fn min(&mut self) -> i32 {
        match self.list.front() {
            None => panic!("No min in empty list"),
            Some(node) => node.min_below
        }
    }
}

// Problem 3.3: Stack of Plates
// Imagine a (literal) stack of plates. If the stack gets too high, it might topple. Therefore, in real life, we would likely start a new stack when the previous stack exceeds some threshold. Implement a data structure SetOfStacks that mimics this. SetOfStacks should be composed of several stacks and should create a new stack once the previous one exceeds capacity. SetOfStacks.push() and SetOfStacks.pop() should behave identically to a single stack (that is, pop() should return the same values as it would if there were just a single stack).

#[derive(Debug)]
pub struct StackOPlates {
    stacks: Vec<Vec<i32>>,
    threshold: usize,
}

impl StackOPlates {
    pub fn new(threshold: usize) -> Self {
        StackOPlates { stacks: Vec::new(), threshold: threshold }
    }

    pub fn push(&mut self, value: i32) {
        for stack in self.stacks.iter_mut() {
            // Since any stack may have had a plate removed, iterate from first to last
            if stack.len() < self.threshold {
                stack.push(value);
                return;
            }
        }
        // Empty stacks or all full
        let mut new_list = Vec::new();
        new_list.push(value);
        self.stacks.push(new_list);
    }

    pub fn pop(&mut self) -> i32 {
        match self.stacks.last_mut() {
            None => panic!("Can't pop from empty stack"),
            Some(stack) => {
                match stack.pop() {
                    None => panic!("Front stack is empty!"),
                    Some(value) => {
                        if stack.is_empty() {
                            // Delete front stack
                            self.stacks.pop();
                        }
                        value
                    }
                }
            }
        }
    }

    pub fn pop_kth(&mut self, k: usize) -> i32 {
        match self.stacks.get_mut(k) {
            None => panic!("No {k}th stack"),
            Some(stack) => match stack.pop() {
                None => panic!("{k}th stack is empty"),
                Some(value) => value
            }
        }
    }
}

// Problem 3.4: Queue via Stacks
// Implement a MyQueue class which implements a queue using two stacks.

pub struct MyQueue {
    stack_push: Vec<i32>,
    stack_pop: Vec<i32>,
}

impl MyQueue {
    pub fn new() -> Self {
        MyQueue { stack_push: Vec::new(), stack_pop: Vec::new() }
    }

    pub fn push(&mut self, value: i32) {
        self.stack_push.push(value);
    }

    pub fn pop(&mut self) -> i32 {
        match self.stack_pop.pop() {
            None => {
                // Revert the push stack into the pop stack
                while !self.stack_push.is_empty() {
                    self.stack_pop.push(self.stack_push.pop().unwrap());
                }
                match self.stack_pop.pop() {
                    None => panic!("Queue is empty"),
                    Some(value) => value
                }
            },
            Some(value) => {
                value
            }
        }
    }
}

// Problem 3.5: Sort Stack
// Write a program to sort a stack in ascending order. You should not make any assumptions about how the stack is implemented. The following are the only functions that should be used to write this program: push | pop | peek | isEmpty.

pub fn sort_stack_3_5(stack: &mut Vec<i32>) {
    // We want to sort our stack so that its smallest elements are at the top
    // We use a secondary stack sorted the other way
    let mut inverted_stack = Vec::new();
    while let Some(value) = stack.pop() {
        while let Some(&top) = inverted_stack.last() {
            if top < value {
                stack.push(inverted_stack.pop().unwrap());
            } else {
                break;
            }
        }
        inverted_stack.push(value);
    }
    // The inverted stack now contain the full stack sorted in reverse order
    while let Some(value) = inverted_stack.pop() {
        stack.push(value);
    }
}

// Problem 3.6: Animal Shelter
// An animal shelter, which holds only dogs and cats, operates on a strictly "first in, first out" basis. People must adopt either the "oldest" (based on arrival time) of all animals at the shelter, or they can select whether they would prefer a dog or a cat (and will receive the oldest animal of that type). They cannot select which specific animal they would like. Create the data structures to maintain this system and
// implement operations such as enqueue, dequeueAny, dequeueDog, and dequeueCat. You may use the built-in LinkedList class.

#[derive(Debug, PartialEq)]
pub enum Animal {
    Cat,
    Dog,
}

#[derive(Debug)]
pub struct AnimalEntry {
    index: usize,
}

#[derive(Debug)]
pub struct AnimalShelter36 {
    cat_stack: Vec<AnimalEntry>,
    dog_stack: Vec<AnimalEntry>,
    next_index: usize,
}

impl AnimalShelter36 {
    pub fn new() -> Self {
        AnimalShelter36 { cat_stack: Vec::new(), dog_stack: Vec::new(), next_index: 0 }
    }

    pub fn enqueue(&mut self, animal: Animal) {
        let animal_entry = AnimalEntry { index: self.next_index };
        self.next_index += 1;
        match animal {
            Animal::Cat => self.cat_stack.push(animal_entry),
            Animal::Dog => self.dog_stack.push(animal_entry),
        }
    }

    pub fn dequeue_any(&mut self) -> Option<Animal> {
        match (self.cat_stack.last(), self.dog_stack.last()) {
            (Some(cat), Some(dog)) => {
                if cat.index < dog.index {
                    self.dequeue_dog()
                } else {
                    self.dequeue_cat()
                }
            },
            (Some(_), None) => self.dequeue_cat(),
            (None, Some(_)) => self.dequeue_dog(),
            (None, None) => None,
        }
    }

    pub fn dequeue_cat(&mut self) -> Option<Animal> {
        match self.cat_stack.pop() {
            None => None,
            Some(_) => Some(Animal::Cat),
        }
    }

    pub fn dequeue_dog(&mut self) -> Option<Animal> {
        match self.dog_stack.pop() {
            None => return None,
            Some(_) => return Some(Animal::Dog),
        }
    }
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

    #[test]
    fn test_string_compression_1_6() {
        let test_string = String::from("aabcccccaaa");
        let result = string_compression_1_6(test_string);
        assert_eq!(result, "a2b1c5a3");
        let test_string = String::from("abc");
        let result = string_compression_1_6(test_string);
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_rotate_matrix_1_7() {
        let mut matrix = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ];
        rotate_matrix_1_7(&mut matrix);
        let expected_result = vec![
            vec![7, 4, 1],
            vec![8, 5, 2],
            vec![9, 6, 3],
        ];
        assert_eq!(matrix, expected_result);
    }

    #[test]
    fn test_zero_matrix_1_8() {
        let mut matrix = vec![
            vec![1, 2, 3],
            vec![4, 0, 6],
            vec![7, 8, 9],
        ];
        zero_matrix_1_8(&mut matrix);
        let expected_result = vec![
            vec![1, 0, 3],
            vec![0, 0, 0],
            vec![7, 0, 9],
        ];
        assert_eq!(matrix, expected_result);
    }

    #[test]
    fn test_is_rotation_1_9() {
        let test_string_a = String::from("waterbottle");
        let test_string_b = String::from("erbottlewat");
        let result = is_rotation_1_9(test_string_a, test_string_b);
        assert_eq!(result, true);
        let test_string_a = String::from("waterbottle");
        let test_string_b = String::from("bottlewater");
        let result = is_rotation_1_9(test_string_a, test_string_b);
        assert_eq!(result, true);
        let test_string_a = String::from("waterbottle");
        let test_string_b = String::from("erbottlewas");
        let result = is_rotation_1_9(test_string_a, test_string_b);
        assert_eq!(result, false);
    }

    #[test]
    fn test_remove_duplicate_2_1() {
        let mut list = SinglyLinkedList::new();
        list.append(1);
        list.append(1);
        list.append(2);
        list.append(2);
        list.append(3);
        list.append(3);
        // list.remove_duplicates();
        remove_duplicate_2_1(&mut list);

        assert_eq!(list.to_vector(), vec![1, 2, 3]);
    }

    #[test]
    fn test_kth_to_last_2_2() {
        let mut list = SinglyLinkedList::new();
        list.append(1);
        list.append(2);
        list.append(3);
        list.append(4);
        list.append(5);

        let result = kth_to_last_2_2(&mut list, 2);
        assert_eq!(result, Ok(4));
    }

    #[test]
    fn test_delete_middle_node_2_3() {
        let mut list = SinglyLinkedList::new();
        list.append(1);
        list.append(2);
        list.append(3);
        list.append(4);
        list.append(5);

        let mut node_to_delete = list.head.as_mut().unwrap().next.as_mut().unwrap().next.as_mut().unwrap();
        delete_middle_node_2_3(&mut node_to_delete);

        assert_eq!(node_to_delete.value, 4);
        assert_eq!(node_to_delete.next.as_mut().unwrap().value, 5);
        assert_eq!(list.to_vector(), vec![1, 2, 4, 5]);
    }

    #[test]
    fn test_partition_2_4() {
        let mut list = SinglyLinkedList::new();
        list.append(3);
        list.append(5);
        list.append(8);
        list.append(5);
        list.append(10);
        list.append(2);
        list.append(1);

        partition_2_4(&mut list, 5);

        assert_eq!(list.to_vector(), vec![3, 2, 1, 5, 8, 5, 10]);

    }

    #[test]
    fn test_sum_lists_1s_first_2_5() {
        let mut list_a: SinglyLinkedList<u8> = SinglyLinkedList::new();
        list_a.append(7);
        list_a.append(1);
        list_a.append(6);
        let mut list_b: SinglyLinkedList<u8> = SinglyLinkedList::new();
        list_b.append(5);
        list_b.append(9);
        list_b.append(2);
        let result = sum_lists_1s_first_2_5(&list_a, &list_b);
        assert_eq!(result.to_vector(), vec![2, 1, 9]);
    }

    #[test]
    fn test_sum_lists_1s_last_2_5() {
        let mut list_a: SinglyLinkedList<u8> = SinglyLinkedList::new();
        list_a.append(6);
        list_a.append(1);
        list_a.append(7);
        let mut list_b: SinglyLinkedList<u8> = SinglyLinkedList::new();
        list_b.append(2);
        list_b.append(9);
        list_b.append(5);
        let result = sum_lists_1s_last_2_5(&list_a, &list_b);
        assert_eq!(result.to_vector(), vec![9, 1, 2]);
    }

    #[test]
    fn test_palindrome_2_6() {
        let mut list = SinglyLinkedList::new();
        list.append(1);
        list.append(2);
        list.append(3);
        list.append(2);
        list.append(1);
        let result = palindrome_2_6(list);
        assert_eq!(result, true);

        let mut list = SinglyLinkedList::new();
        list.append(1);
        list.append(2);
        list.append(2);
        list.append(1);
        let result = palindrome_2_6(list);
        assert_eq!(result, true);

        let mut list = SinglyLinkedList::new();
        list.append(1);
        list.append(2);
        list.append(3);
        list.append(4);
        list.append(5);
        let result = palindrome_2_6(list);
        assert_eq!(result, false);
    }

    #[test]
    fn test_intersection_2_7() {
        let mut list_a = RcSinglyLinkedList::new();
        list_a.append(1);
        list_a.append(2);
        list_a.append(3);

        let mut list_b = RcSinglyLinkedList::new();
        list_b.append(4);
        list_b.append(5);
        list_b.append(6);

        assert!(intersection_2_7(&list_a, &list_b).is_none());

        // Create an intersection
        let mut current_a = list_a.head.clone();
        for _ in 0..2 {
            current_a = current_a.unwrap().borrow().next.clone();
        }
        let intersecting_node = current_a.unwrap();
        let mut current_b = list_b.head.clone();
        for _ in 0..2 {
            current_b = current_b.unwrap().borrow().next.clone();
        }
        current_b.unwrap().borrow_mut().next = Some(intersecting_node.clone());

        let result = intersection_2_7(&list_a, &list_b);
        assert!(result.is_some() && Rc::ptr_eq(result.as_ref().unwrap(), &intersecting_node));
    }

    #[test]
    fn test_loop_detection_2_8() {
        let mut list = RcSinglyLinkedList::new();
        list.append(1);
        list.append(2);
        list.append(3);
        let loop_start = Rc::new(RefCell::new(RcNode { value: 4, next: None }));
        list.head.as_ref().unwrap().borrow_mut().next = Some(loop_start.clone());
        loop_start.borrow_mut().next = Some(Rc::new(RefCell::new(RcNode { value: 5, next: Some(loop_start.clone()) })));

        let result = loop_detection_2_8(&list);
        assert!(result.is_some() && Rc::ptr_eq(result.as_ref().unwrap(), &loop_start));
    }

    #[test]
    fn test_stacks_on_array_push_and_pop() {
        let mut stacks = StacksOnArray::new(3);
        stacks.push(0, 10);
        stacks.push(0, 20);
        stacks.push(1, 30);
        stacks.push(2, 40);

        assert_eq!(stacks.pop(0), 20);
        assert_eq!(stacks.pop(0), 10);
        assert_eq!(stacks.pop(1), 30);
        assert_eq!(stacks.pop(2), 40);
    }

    #[test]
    fn test_stacks_on_array_peek() {
        let mut stacks = StacksOnArray::new(3);
        stacks.push(0, 10);
        stacks.push(0, 20);
        stacks.push(1, 30);

        assert_eq!(stacks.peek(0), 20);
        assert_eq!(stacks.peek(1), 30);
    }

    #[test]
    fn test_stacks_on_array_is_empty() {
        let mut stacks = StacksOnArray::new(3);
        assert!(stacks.is_empty(0));
        stacks.push(0, 10);
        assert!(!stacks.is_empty(0));
    }

    #[test]
    fn test_stacks_on_array_is_full() {
        let mut stacks = StacksOnArray::new(2);
        stacks.push(0, 10);
        stacks.push(0, 20);
        assert!(stacks.is_full(0));
    }

    #[test]
    #[should_panic(expected = "Invalid stack id 3")]
    fn test_stacks_on_array_invalid_stack_id_push() {
        let mut stacks = StacksOnArray::new(3);
        stacks.push(3, 10);
    }

    #[test]
    #[should_panic(expected = "Invalid stack id 3")]
    fn test_stacks_on_array_invalid_stack_id_pop() {
        let mut stacks: StacksOnArray<i32> = StacksOnArray::new(3);
        stacks.pop(3);
    }

    #[test]
    #[should_panic(expected = "Invalid stack id 3")]
    fn test_stacks_on_array_invalid_stack_id_peek() {
        let stacks: StacksOnArray<i32> = StacksOnArray::new(3);
        stacks.peek(3);
    }

    #[test]
    #[should_panic(expected = "Stack 0 is full")]
    fn test_stacks_on_array_push_to_full_stack() {
        let mut stacks = StacksOnArray::new(2);
        stacks.push(0, 10);
        stacks.push(0, 20);
        stacks.push(0, 30);
    }

    #[test]
    #[should_panic(expected = "Stack 0 is empty")]
    fn test_stacks_on_array_pop_from_empty_stack() {
        let mut stacks: StacksOnArray<i32> = StacksOnArray::new(3);
        stacks.pop(0);
    }

    #[test]
    #[should_panic(expected = "Stack 0 is empty")]
    fn test_stacks_on_array_peek_empty_stack() {
        let stacks: StacksOnArray<i32> = StacksOnArray::new(3);
        stacks.peek(0);
    }

    #[test]
    fn test_stack_w_min_push_and_pop() {
        let mut stack = StackWMin::new();
        stack.push(5);
        stack.push(3);
        stack.push(7);

        assert_eq!(stack.pop(), 7);
        assert_eq!(stack.pop(), 3);
        assert_eq!(stack.pop(), 5);
    }

    #[test]
    fn test_stack_w_min_min() {
        let mut stack = StackWMin::new();
        stack.push(5);
        assert_eq!(stack.min(), 5);
        stack.push(3);
        assert_eq!(stack.min(), 3);
        stack.push(7);
        assert_eq!(stack.min(), 3);
        stack.pop();
        assert_eq!(stack.min(), 3);
        stack.pop();
        assert_eq!(stack.min(), 5);
    }

    #[test]
    #[should_panic(expected = "Cannot pop from empty list")]
    fn test_stack_w_min_pop_empty_stack() {
        let mut stack = StackWMin::new();
        stack.pop();
    }

    #[test]
    #[should_panic(expected = "No min in empty list")]
    fn test_stack_w_min_min_empty_stack() {
        let mut stack = StackWMin::new();
        stack.min();
    }

    #[test]
    fn test_stack_w_min_push_same_min() {
        let mut stack = StackWMin::new();
        stack.push(5);
        stack.push(5);
        stack.push(5);

        assert_eq!(stack.min(), 5);
        stack.pop();
        assert_eq!(stack.min(), 5);
        stack.pop();
        assert_eq!(stack.min(), 5);
    }

    #[test]
    fn test_stack_w_min_push_decreasing_order() {
        let mut stack = StackWMin::new();
        stack.push(5);
        stack.push(4);
        stack.push(3);
        stack.push(2);
        stack.push(1);

        assert_eq!(stack.min(), 1);
        stack.pop();
        assert_eq!(stack.min(), 2);
        stack.pop();
        assert_eq!(stack.min(), 3);
        stack.pop();
        assert_eq!(stack.min(), 4);
        stack.pop();
        assert_eq!(stack.min(), 5);
    }

    #[test]
    fn test_stack_w_min_push_increasing_order() {
        let mut stack = StackWMin::new();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        stack.push(4);
        stack.push(5);

        assert_eq!(stack.min(), 1);
        stack.pop();
        assert_eq!(stack.min(), 1);
        stack.pop();
        assert_eq!(stack.min(), 1);
    }

    #[test]
    fn test_stack_o_plates_push_and_pop() {
        let mut stack = StackOPlates::new(3);
        stack.push(1);
        stack.push(2);
        stack.push(3);
        stack.push(4); // Should create a new stack
        stack.push(5);

        assert_eq!(stack.pop(), 5);
        assert_eq!(stack.pop(), 4);
        assert_eq!(stack.pop(), 3);
        assert_eq!(stack.pop(), 2);
        assert_eq!(stack.pop(), 1);
    }

    #[test]
    #[should_panic(expected = "Can't pop from empty stack")]
    fn test_stack_o_plates_pop_empty_stack() {
        let mut stack = StackOPlates::new(3);
        stack.pop();
    }

    #[test]
    fn test_stack_o_plates_multiple_stacks() {
        let mut stack = StackOPlates::new(2);
        stack.push(1);
        stack.push(2);
        stack.push(3); // New stack
        stack.push(4);
        stack.push(5); // Another new stack

        assert_eq!(stack.pop(), 5);
        assert_eq!(stack.pop(), 4);
        assert_eq!(stack.pop(), 3);
        assert_eq!(stack.pop(), 2);
        assert_eq!(stack.pop(), 1);
    }

    #[test]
    fn test_stack_o_plates_push_pop_interleaved() {
        let mut stack = StackOPlates::new(2);
        stack.push(1);
        stack.push(2);
        stack.push(3); // New stack
        assert_eq!(stack.pop(), 3);
        stack.push(4);
        stack.push(5); // New stack
        assert_eq!(stack.pop(), 5);
        assert_eq!(stack.pop(), 4);
        assert_eq!(stack.pop(), 2);
        assert_eq!(stack.pop(), 1);
    }

    #[test]
    fn test_stack_o_plates_empty_after_pop() {
        let mut stack = StackOPlates::new(2);
        stack.push(1);
        stack.push(2);
        stack.push(3); // New stack
        assert_eq!(stack.pop(), 3);
        assert_eq!(stack.pop(), 2);
        assert_eq!(stack.pop(), 1);

        // Stack should now be empty
        assert!(stack.stacks.is_empty());
    }

    #[test]
    fn test_stack_o_plates_pop_kth() {
        let mut stack = StackOPlates::new(2);
        stack.push(1);
        stack.push(2);
        stack.push(3); // New stack
        stack.push(4);
        stack.push(5); // Another new stack

        assert_eq!(stack.pop_kth(0), 2); // Pop from the first stack
        assert_eq!(stack.pop_kth(1), 4); // Pop from the second stack
        assert_eq!(stack.pop_kth(1), 3); // Pop the remaining element from the second stack
        assert_eq!(stack.pop_kth(0), 1); // Pop the remaining element from the first stack
        assert_eq!(stack.pop_kth(2), 5); // Pop from the last stack
    }

    #[test]
    fn test_stack_o_plates_pop_kth_empty_middle_stack() {
        let mut stack = StackOPlates::new(2);
        stack.push(1);
        stack.push(2);
        stack.push(3); // New stack
        stack.push(4);
        stack.push(5); // Another new stack

        assert_eq!(stack.pop_kth(1), 4); // Pop from the second stack
        assert_eq!(stack.pop_kth(1), 3); // Pop the remaining element from the second stack

        // Push more elements to the now-empty middle stack
        stack.push(6);
        stack.push(7);

        assert_eq!(stack.pop_kth(1), 7); // Pop from the middle stack
        assert_eq!(stack.pop_kth(1), 6); // Pop the remaining element from the middle stack
    }

    #[test]
    #[should_panic(expected = "No 3th stack")]
    fn test_stack_o_plates_pop_kth_invalid_index() {
        let mut stack = StackOPlates::new(2);
        stack.push(1);
        stack.push(2);
        stack.push(3); // New stack

        stack.pop_kth(3); // Invalid index
    }

    #[test]
    #[should_panic(expected = "1th stack is empty")]
    fn test_stack_o_plates_pop_kth_empty_stack() {
        let mut stack = StackOPlates::new(2);
        stack.push(1);
        stack.push(2);
        stack.push(3); // New stack

        stack.pop_kth(1); // Pop from the second stack
        stack.pop_kth(1); // Pop the remaining element from the second stack
        stack.pop_kth(1); // Attempt to pop from an empty stack
    }

    #[test]
    fn test_myqueue_push_and_pop() {
        let mut queue = MyQueue::new();
        queue.push(1);
        queue.push(2);
        queue.push(3);

        assert_eq!(queue.pop(), 1);
        assert_eq!(queue.pop(), 2);
        assert_eq!(queue.pop(), 3);
    }

    #[test]
    #[should_panic(expected = "Queue is empty")]
    fn test_myqueue_pop_empty_queue() {
        let mut queue = MyQueue::new();
        queue.pop();
    }

    #[test]
    fn test_myqueue_push_pop_interleaved() {
        let mut queue = MyQueue::new();
        queue.push(1);
        assert_eq!(queue.pop(), 1);
        queue.push(2);
        queue.push(3);
        assert_eq!(queue.pop(), 2);
        queue.push(4);
        assert_eq!(queue.pop(), 3);
        assert_eq!(queue.pop(), 4);
    }

    #[test]
    fn test_myqueue_multiple_push_and_pop() {
        let mut queue = MyQueue::new();
        for i in 1..=10 {
            queue.push(i);
        }
        for i in 1..=10 {
            assert_eq!(queue.pop(), i);
        }
    }

    #[test]
    fn test_myqueue_empty_after_pop() {
        let mut queue = MyQueue::new();
        queue.push(1);
        queue.push(2);
        queue.pop();
        queue.pop();

        // Queue should now be empty
        assert!(queue.stack_push.is_empty());
        assert!(queue.stack_pop.is_empty());
    }

    #[test]
    fn test_myqueue_push_after_emptying() {
        let mut queue = MyQueue::new();
        queue.push(1);
        queue.push(2);
        queue.pop();
        queue.pop();

        queue.push(3);
        queue.push(4);

        assert_eq!(queue.pop(), 3);
        assert_eq!(queue.pop(), 4);
    }

    #[test]
    fn test_myqueue_large_number_of_elements_intertwined() {
        let mut queue = MyQueue::new();
        for i in 1..=500 {
            queue.push(i);
        }
        for i in 1..=500 {
            assert_eq!(queue.pop(), i);
        }
        for i in 1..=500 {
            queue.push(i);
        }
        for i in 1..=500 {
            assert_eq!(queue.pop(), i);
        }
    }

    #[test]
    fn test_myqueue_push_pop_with_duplicates() {
        let mut queue = MyQueue::new();
        queue.push(1);
        queue.push(1);
        queue.push(2);
        queue.push(2);

        assert_eq!(queue.pop(), 1);
        assert_eq!(queue.pop(), 1);
        assert_eq!(queue.pop(), 2);
        assert_eq!(queue.pop(), 2);
    }

    #[test]
    fn test_myqueue_push_pop_single_element() {
        let mut queue = MyQueue::new();
        queue.push(42);
        assert_eq!(queue.pop(), 42);
    }

    #[test]
    fn test_sort_stack_3_5() {
        let mut stack = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        sort_stack_3_5(&mut stack);
        assert_eq!(stack, vec![1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]);

        let mut stack = vec![10, 20, 30, 40, 50];
        sort_stack_3_5(&mut stack);
        assert_eq!(stack, vec![10, 20, 30, 40, 50]);

        let mut stack = vec![50, 40, 30, 20, 10];
        sort_stack_3_5(&mut stack);
        assert_eq!(stack, vec![10, 20, 30, 40, 50]);

        let mut stack = vec![1];
        sort_stack_3_5(&mut stack);
        assert_eq!(stack, vec![1]);

        let mut stack: Vec<i32> = vec![];
        sort_stack_3_5(&mut stack);
        assert_eq!(stack, vec![]);
    }

    #[test]
    fn test_animal_shelter_enqueue_and_dequeue_any() {
        let mut shelter = AnimalShelter36::new();
        shelter.enqueue(Animal::Cat);
        shelter.enqueue(Animal::Dog);
        shelter.enqueue(Animal::Cat);

        assert_eq!(shelter.dequeue_any(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_any(), Some(Animal::Dog));
        assert_eq!(shelter.dequeue_any(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_any(), None);
    }

    #[test]
    fn test_animal_shelter_dequeue_cat() {
        let mut shelter = AnimalShelter36::new();
        shelter.enqueue(Animal::Cat);
        shelter.enqueue(Animal::Dog);
        shelter.enqueue(Animal::Cat);

        assert_eq!(shelter.dequeue_cat(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_cat(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_cat(), None);
    }

    #[test]
    fn test_animal_shelter_dequeue_dog() {
        let mut shelter = AnimalShelter36::new();
        shelter.enqueue(Animal::Dog);
        shelter.enqueue(Animal::Cat);
        shelter.enqueue(Animal::Dog);

        assert_eq!(shelter.dequeue_dog(), Some(Animal::Dog));
        assert_eq!(shelter.dequeue_dog(), Some(Animal::Dog));
        assert_eq!(shelter.dequeue_dog(), None);
    }

    #[test]
    fn test_animal_shelter_mixed_operations() {
        let mut shelter = AnimalShelter36::new();
        shelter.enqueue(Animal::Cat);
        shelter.enqueue(Animal::Dog);
        shelter.enqueue(Animal::Cat);
        shelter.enqueue(Animal::Dog);

        assert_eq!(shelter.dequeue_any(), Some(Animal::Dog));
        assert_eq!(shelter.dequeue_any(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_dog(), Some(Animal::Dog));
        assert_eq!(shelter.dequeue_cat(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_any(), None);
    }

    #[test]
    fn test_animal_shelter_empty_dequeues() {
        let mut shelter = AnimalShelter36::new();

        assert_eq!(shelter.dequeue_any(), None);
        assert_eq!(shelter.dequeue_cat(), None);
        assert_eq!(shelter.dequeue_dog(), None);
    }

    #[test]
    fn test_animal_shelter_enqueue_only_cats() {
        let mut shelter = AnimalShelter36::new();
        shelter.enqueue(Animal::Cat);
        shelter.enqueue(Animal::Cat);

        assert_eq!(shelter.dequeue_any(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_any(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_any(), None);
    }

    #[test]
    fn test_animal_shelter_enqueue_only_dogs() {
        let mut shelter = AnimalShelter36::new();
        shelter.enqueue(Animal::Dog);
        shelter.enqueue(Animal::Dog);

        assert_eq!(shelter.dequeue_any(), Some(Animal::Dog));
        assert_eq!(shelter.dequeue_any(), Some(Animal::Dog));
        assert_eq!(shelter.dequeue_any(), None);
    }

    #[test]
    fn test_animal_shelter_enqueue_and_dequeue_order() {
        let mut shelter = AnimalShelter36::new();
        shelter.enqueue(Animal::Dog);
        shelter.enqueue(Animal::Cat);
        shelter.enqueue(Animal::Dog);
        shelter.enqueue(Animal::Cat);

        assert_eq!(shelter.dequeue_any(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_any(), Some(Animal::Dog));
        assert_eq!(shelter.dequeue_any(), Some(Animal::Cat));
        assert_eq!(shelter.dequeue_any(), Some(Animal::Dog));
    }
}
