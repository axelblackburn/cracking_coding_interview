use rand::Rng;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::LinkedList;
use std::collections::VecDeque;
use std::ptr::null_mut;
use std::rc::Rc;
use std::rc::Weak;

// Quicksort

fn quicksort_helper_recursive(slice: &mut [i32]) {
    let len = slice.len();
    if len <= 1 {
        return;
    }

    let partition_index = quicksort_helper_partition(slice);

    let (left, right) = slice.split_at_mut(partition_index);
    quicksort_helper_recursive(left);
    quicksort_helper_recursive(&mut right[1..]);
}

fn quicksort_helper_partition(slice: &mut [i32]) -> usize {
    let mut rng = rand::rng();
    let len = slice.len();
    let pivot_random_index = rng.random_range(0..len);
    slice.swap(pivot_random_index, len - 1);

    let pivot = slice[len - 1];
    let mut pivot_index = 0;

    for i in 0 .. len - 1 {
        if slice[i] <= pivot {
            slice.swap(pivot_index, i);
            pivot_index += 1;
        }
    }

    slice.swap(pivot_index, len - 1);

    pivot_index
}

pub fn quicksort(list: &mut [i32]) {
    quicksort_helper_recursive(list);
}

// Merge sort

fn mergesort_helper_merge(output: &mut [i32], left_buffer: &[i32], right_buffer: &[i32]) {
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;

    while i < left_buffer.len() && j < right_buffer.len() {
        if left_buffer[i] <= right_buffer[j] {
            output[k] = left_buffer[i];
            i += 1;
        } else {
            output[k] = right_buffer[j];
            j += 1;
        }
        k += 1;
    }

    while i < left_buffer.len() {
        output[k] = left_buffer[i];
        i += 1;
        k += 1;
    }

    while j < right_buffer.len() {
        output[k] = right_buffer[j];
        j += 1;
        k += 1;
    }
}

// Top Down
fn mergesort_topdown_helper_split(input: &mut [i32], buffer: &mut [i32]) {
    let len = input.len();
    if len <= 1 {
        return;
    }

    let index_middle = len / 2;
    mergesort_topdown_helper_split(&mut buffer[..index_middle], &mut input[..index_middle]);
    mergesort_topdown_helper_split(&mut buffer[index_middle..], &mut input[index_middle..]);
    mergesort_helper_merge(input, &buffer[..index_middle], &buffer[index_middle..]);
}


pub fn mergesort_topdown(input: &mut [i32]) {
    let mut buffer = input.to_vec();
    mergesort_topdown_helper_split(input, &mut buffer);
}

// Bottom Up
pub fn mergesort_bottomup(input: &mut [i32]) {
    let len = input.len();
    if len <= 1 {
        return;
    }

    let mut buffer = input.to_vec();
    let mut source = input;
    let mut dest = &mut buffer[..];

    let mut width = 1;
    while width < len {
        let mut i = 0;
        while i < len {
            let middle = len.min(i + width);
            let end = len.min(i + 2 * width);
            mergesort_helper_merge(&mut dest[i..end], &source[i..middle], &source[middle..end]);
            i += 2 * width;
        }

        std::mem::swap(&mut source, &mut dest);
        width *= 2;
    }
}

// Heap sort

fn heapsort_helper_heapify(input: &mut [i32]) {
    let len = input.len();
    if len <= 1 {
        return;
    }

    for i in (0..len / 2).rev() {
        heapsort_helper_sift_down(input, i, len);
    }
}

fn heapsort_helper_sift_down(input: &mut [i32], mut root: usize, end: usize) {
    loop {
        let left = 2 * root + 1;
        let right = 2 * root + 2;
        let mut largest = root;

        if left < end && input[left] > input[largest] {
            largest = left;
        }
        if right < end && input[right] > input[largest] {
            largest = right;
        }
        if largest == root {
            break;
        }

        input.swap(root, largest);
        root = largest;
    }
}

pub fn heapsort(input: &mut [i32]) {
    let len = input.len();
    heapsort_helper_heapify(input);

    for end in (1..len).rev() {
        input.swap(0, end);
        heapsort_helper_sift_down(input, 0, end);
    }
}

// Amazon coding example: num ways
// Give a number of steps N, and knowing that one can only go up 1 or 2 steps at a time, return the number of ways one can climb the stairs

// Idea: recursive solution
pub fn amazon_num_ways(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }

    // Highly inefficient
    // return amazon_num_ways(n - 2) + amazon_num_ways(n - 1);

    let mut last_two: [usize; 2] = [1, 1];
    for _ in 2..n {
        let three_ago = last_two[0];
        last_two[0] = last_two[1];
        last_two[1] = last_two[0] + three_ago;
    }

    last_two.iter().sum()
}

// Now with an arbitrary list of possible steps at a time
pub fn amazon_num_ways_any_steps(n: usize, steps: &Vec<usize>) -> usize {
    // Assumption: steps is sorted, we could sort it otherwise
    if n < steps[0] {
        return 0;
    }

    // Iterative solution saves space but is highly ineficient
    // let previous_stop = | step: &usize | {
    //     amazon_num_ways_any_steps(n.saturating_sub(*step), steps)
    // };

    // for i in 0..steps.len() {
    //     if n == steps[i] {
    //         return 1 + steps.iter().map(previous_stop).sum::<usize>();
    //     }
    // }

    // steps.iter().map(previous_stop).sum()
    let mut results = vec![0; n+1];
    // Assume that there is always a way, unlike the recursive solution above
    results[0] = 1;
    for i in 1..=n {
        let mut total = 0;
        for j in steps {
            if i >= *j {
                total += results[i - j];
            }
        }
        results[i] = total;
    }

    results[n]
}

// Daily Coding Problem 2025/06/01 EASY
// Given a list of numbers and a number k, return whether any two numbers from the list add up to k.
// For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.
// Bonus: Can you do this in one pass?

pub fn daily_easy_2025_06_01(list: &[i32], k: i32) -> bool {
    // One pass: create a HashSet of k-n for each n encountered
    let mut k_minus_i_set = HashSet::new();
    for n in list {
        if k_minus_i_set.contains(n) {
            return true;
        }
        k_minus_i_set.insert(k - n);
    }

    false
}

// Daily Coding Problem 2025/06/01 HARD
// Given an array of integers, return a new array such that
// each element at index i of the new array is the product of
// all the numbers in the original array except the one at i.
// For example, if our input was [1, 2, 3, 4, 5], the expected output would be
// [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would be [2, 3, 6].
// Follow-up: what if you can't use division?

pub fn daily_hard_2025_06_01(list: &[i32]) -> Vec<i32> {
    // With division: compute the total product then divide each element when constructing the new array
    let mut total_multiplication = 1;
    // Assumption: no 0 element
    list.iter().for_each(|n| total_multiplication *= n);
    let mut result = vec![0; list.len()];
    for i in 0..list.len() {
        result[i] = total_multiplication / list[i];
    }
    result
}

pub fn daily_hard_2025_06_01_no_div(list: &[i32]) -> Vec<i32> {
    let mut result = vec![1; list.len()];
    let mut multiplier = 1;
    for i in 0..list.len() {
        result[i] = multiplier;
        multiplier *= list[i];
    }

    let mut multiplier = 1;
    for i in (0..list.len()).rev() {
        result[i] *= multiplier;
        multiplier *= list[i];
    }

    result
}

// Daily Coding Problem 2025/06/01 HARD explained
// Return a new sorted merged list from K sorted lists, each with size N

pub fn daily_harder_2025_06_01_nk(lists: &[Vec<i32>]) -> Vec<i32> {
    if lists.len() == 0 {
        return Vec::new();
    }

    // Merge sort idea: have 1 index per list and merge
    let n = lists[0].len();
    let mut result = Vec::with_capacity(n * lists.len());
    let mut indexes = vec![0; lists.len()];

    while result.len() < n * lists.len() {
        // Pick the index to increment by comparing its element with all the others'
        let mut min: Option<i32> = None;
        let mut min_index: Option<usize> = None;

        for i in 0..indexes.len() {
            let index = indexes[i];
            if index < n {
                if min_index.is_none() || lists[i][index] < min.unwrap() {
                    min = Some(lists[i][index]);
                    min_index = Some(i);
                }
            }
        }

        if let (Some(val), Some(i)) = (min, min_index) {
            result.push(val);
            indexes[i] += 1;
        }
    }

    result
}

pub fn daily_harder_2025_06_01_heapmin(lists: &[Vec<i32>]) -> Vec<i32> {
    if lists.is_empty() {
        return Vec::new();
    }

    let mut heap = BinaryHeap::new();
    let mut indexes = vec![0; lists.len()];
    let n = lists[0].len();
    let mut result = Vec::with_capacity(n * lists.len());

    for i in 0..lists.len() {
        heap.push((Reverse(lists[i][0]), i));
    }

    while result.len() < n * lists.len() {
        if let Some((value, i)) = heap.pop() {
            result.push(value.0);
            indexes[i] += 1;
            if indexes[i] < n {
                heap.push((Reverse(lists[i][indexes[i]]), i));
            }
        }
    }

    result
}

// Given the root to a binary tree, implement serialize(root), which serializes the tree into a string, and deserialize(s), which deserializes the string back into the tree.
// For example, given the following Node class
// class Node:
//    def __init__(self, val, left=None, right=None):
//        self.val = val
//        self.left = left
//        self.right = right
// The following test should pass:
// node = Node('root', Node('left', Node('left.left')), Node('right'))
// assert deserialize(serialize(node)).left.left.val == 'left.left'

#[derive(Debug, PartialEq)]
pub struct BinaryNode {
    pub val: String,
    pub left: Option<Box<BinaryNode>>,
    pub right: Option<Box<BinaryNode>>,
}

impl BinaryNode {
    pub fn new(val: impl Into<String>, left: Option<Box<BinaryNode>>, right: Option<Box<BinaryNode>>) -> Self {
        BinaryNode {
            val: val.into(),
            left,
            right,
        }
    }
}

pub fn serialize_daily_med_2025_06_02(root: &BinaryNode) -> Result<String, String> {
    if root.val.contains('|') || root.val.contains(']') || root.val.contains('[') {
        return Err(format!("Invalid value {}", root.val));
    }
    let mut result = String::new();
    result.push_str(&format!("[{}|", &root.val));
    result.push_str(&match (&root.left, &root.right) {
        (None, None) => "|".to_string(),
        (Some(left), None) => format!("{}|", &serialize_daily_med_2025_06_02(&left)?),
        (None, Some(right)) => format!("|{}", &serialize_daily_med_2025_06_02(&right)?),
        (Some(left), Some(right)) => {
            format!("{}|{}",
                &serialize_daily_med_2025_06_02(&left)?,
                &serialize_daily_med_2025_06_02(&right)?)
        }
    });
    result.push(']');

    Ok(result)
}

pub fn deserialize_daily_med_2025_06_02(text: &str) -> Result<BinaryNode, String> {
    if text.is_empty() {
        return Err("Invalid input string".to_string());
    }

    if !text.starts_with("[") || !text.ends_with("]") {
        return Err("Missing outer brackets".into());
    }

    // Our node is between the first [ and the last ]
    let inner = &text[1..text.len()-1];
    let mut parts = Vec::new();
    let mut bracket_count = 0; // +1 per [, -1 per ]
    let mut last = 0;
    for (i, c) in inner.char_indices() {
        match c {
            '[' => bracket_count += 1,
            ']' => bracket_count -= 1,
            '|' => if bracket_count == 0 {
                parts.push(&inner[last..i]);
                last = i + 1;
            },
            _ => {}
        }
    }
    parts.push(&inner[last..]);
    if parts.len() != 3 {
        return Err(format!("Invalid format \"{}\"", inner));
    }

    let value = parts[0].to_string();
    let left = if parts[1].is_empty() {
        None
    } else {
        Some(Box::new(deserialize_daily_med_2025_06_02(parts[1])?))
    };
    let right = if parts[2].is_empty() {
        None
    } else {
        Some(Box::new(deserialize_daily_med_2025_06_02(parts[2])?))
    };

    Ok(BinaryNode::new(value, left, right))
}

// Daily Hard 2025/06/03
// Given an array of integers, find the first missing positive integer in linear time and constant space.
// In other words, find the lowest positive integer that does not exist in the array.
// The array can contain duplicates and negative numbers as well.
// For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0] should give 3.
// You can modify the input array in-place.

pub fn first_missing_daily_hard_2025_06_03(input: &mut Vec<i32>) -> u32 {
    // Let's use the array as such: for each index, it represents the integer index+1, meaning it represents [0..=n]
    // As we explore the array, we rearrange it to move each found number to its location, leaving 0 where one is missing.
    // Then we redo a pass to find the smallest index populated with 0.
    let n = input.len();
    for i in 0..n {
        loop {
            let found_value = input[i];
            if found_value <= 0 {
                // No work to be done, just mark it as available
                input[i] = 0;
                break;
            }

            let found_value_index = (found_value - 1) as usize; // Can't underflow
            if found_value_index == i {
                break;
            }

            if found_value_index < n {
                let swapped_value = input[found_value_index];
                input.swap(i, found_value_index);
                if swapped_value == found_value || swapped_value == 0 {
                    // We're done working on i
                    input[i] = 0;
                    break;
                }
            } else {
                // Mark the slot as available
                input[i] = 0;
                break;
            }
        }
    }

    for (i, &x) in input.iter().enumerate() {
        if x == 0 {
            return (i + 1) as u32;
        }
    }
    (n + 1) as u32
}

// Daily Medium 2025/06/04
// cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first and last element of that pair.
// For example, car(cons(3, 4)) returns 3, and cdr(cons(3, 4)) returns 4.
// Given this implementation of cons:
// def cons(a, b):
//    def pair(f):
//        return f(a, b)
//    return pair
// Implement car and cdr.

pub type FuncDaily20250604<A, B, R> = Rc<dyn Fn(Box<dyn Fn(A, B) -> R>) -> R>;

pub fn cons_daily_med_2025_06_04<A: Copy + 'static, B: Copy + 'static>(a: A, b: B) -> FuncDaily20250604<A, B, A> {
    let data = Rc::new((a, b));
    Rc::new(move |f: Box<dyn Fn(A, B) -> A>| {
        let (a, b) = *data;
        f(a, b)
    })
}

pub fn car_daily_med_2025_06_04<A: Copy + 'static, B: Copy + 'static>(pair: &FuncDaily20250604<A, B, A>) -> A {
    pair(Box::new(|a, _| a))
}

pub fn cdr_daily_med_2025_06_04<A: Copy + 'static, B: Copy + 'static>(pair: &FuncDaily20250604<A, B, B>) -> B {
    pair(Box::new(|_, b| b))
}

// https://www.reddit.com/r/cscareerquestions/comments/apu3ni/a_list_of_questions_i_was_asked_at_top_tech/

// Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.
pub fn first_non_repeating(s: &str) -> isize {
    struct Value {
        i: isize,
        single: bool,
    }
    let mut result = -1;

    let mut encountered: HashMap<char, Value> = HashMap::new();
    for (i, c) in s.char_indices() {
        if let Some(value) = encountered.get_mut(&c) {
            value.single = false;
        } else {
            encountered.insert(c, Value { i: i as isize, single: true });
        }
    }

    for v in encountered.values().filter(|&v| v.single) {
        if result == -1 || v.i < result {
            result = v.i;
        }
    }

    result
}

// Given a 2D grid having radio towers in some cells, mountains in some and the rest are empty.
// A radio tower signal can travel in 4 different directions recursively
// until it hits a mountain in that direction. What radio towers can hear each other?

/*
M = Mountain
O = Open
T = Tower

+------+------+------+------+------+
| T    | O    | O    | O    | O    |
| O    | O    | T    | M    | O    |
| T    | M    | O    | T    | M    |
| O    | T    | O    | M    | M    |
| O    | O    | T    | O    | O    |
+------+------+------+------+------+

Directions: no diagonal
Tower's range? Unlimited.

*/

#[derive(PartialEq)]
pub enum Terrain {
    Mountain,
    Open,
    Tower,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Coordinate {
    row: usize,
    col: usize,
}

pub fn who_can_talk_amazon_screen_2025_06_06(topology: &[Vec<Terrain>]) -> Vec<(Coordinate, Coordinate)> {
    use Terrain::*;

    // 1. Find the towers
    // 2. For each tower, look in each direction for another unobstructed
    // Only need to look right and down, up & left would have been found already
    if topology.is_empty() || topology[0].is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut towers = Vec::new();
    let rows = topology.len();
    let cols = topology[0].len();

    for row in 0..rows {
        for col in 0..cols {
            if topology[row][col] == Tower {
                towers.push(Coordinate { row, col });
            }
        }
    }

    for tower in towers {
        let (row, col) = (tower.row, tower.col);

        // Look right
        for c in col + 1..cols {
            match topology[row][c] {
                Mountain => break,
                Tower => {
                    result.push((tower.clone(), Coordinate { row, col: c }));
                    break;
                },
                Open => {},
            }
        }

        // Look down
        for r in row + 1..rows {
            match topology[r][col] {
                Mountain => break,
                Tower => {
                    result.push((tower.clone(), Coordinate { row: r, col }));
                    break;
                },
                Open => {},
            }
        }
    }

    result
}




// Daily problem hard June 5th
/*
An XOR linked list is a more memory efficient doubly linked list.
Instead of each node holding next and prev fields, it holds a field named both,
which is an XOR of the next node and the previous node.
Implement an XOR linked list; it has an add(element) which adds the element to the end,
and a get(index) which returns the node at index.

If using a language that has no pointers (such as Python),
you can assume you have access to get_pointer and
dereference_pointer functions that converts between nodes and memory addresses.
*/

pub struct NodeXOR {
    value: i32,
    both: usize,
}

pub struct ListXOR {
    data: VecDeque<Box<NodeXOR>>,
    head: *mut NodeXOR,
    tail: *mut NodeXOR,
}

impl ListXOR {
    pub fn new() -> Self {
        ListXOR { data: VecDeque::new(), head: null_mut(), tail: null_mut() }
    }

    pub fn is_empty(&self) -> bool {
        self.head.is_null()
    }

    pub fn clear(&mut self) {
        while let Some(_) = self.pop_front() {}
    }

    pub fn front(&self) -> Option<i32> {
        unsafe { self.head.as_ref().map(|node| node.value) }
    }

    pub fn back(&self) -> Option<i32> {
        unsafe { self.tail.as_ref().map(|node| node.value) }
    }

    pub fn push_front(&mut self, value: i32) {
        let new_node = Box::new(NodeXOR { value, both: self.head as usize });
        let new_ptr = &*new_node as *const NodeXOR as *mut NodeXOR;
        if !self.head.is_null() {
            unsafe {
                (*self.head).both ^= new_ptr as usize;
            }
        } else {
            self.tail = new_ptr;
        }
        self.head = new_ptr;
        self.data.push_front(new_node);
    }

    pub fn push_back(&mut self, value: i32) {
        let new_node = Box::new(NodeXOR { value, both: self.tail as usize });
        let new_ptr = &*new_node as *const NodeXOR as *mut NodeXOR;
        if !self.tail.is_null() {
            unsafe {
                (*self.tail).both ^= new_ptr as usize;
            }
        } else {
            self.head = new_ptr;
        }
        self.tail = new_ptr;
        self.data.push_back(new_node);
    }

    pub fn pop_front(&mut self) -> Option<i32> {
        if self.head.is_null() {
            return None;
        }

        unsafe {
            let old_head = self.head;
            let next_addr = (*old_head).both;
            let result = (*old_head).value;

            if next_addr == 0 {
                // Only one element
                self.head = null_mut();
                self.tail = null_mut();
            } else {
                self.head = next_addr as *mut NodeXOR;
                (*self.head).both ^= old_head as usize;
            }

            self.data.pop_front(); // drop the Box<NodeXOR> from VecDeque
            Some(result)
        }
    }

    pub fn pop_back(&mut self) -> Option<i32> {
        if self.tail.is_null() {
            return None;
        }

        unsafe {
            let old_tail = self.tail;
            let prev_addr = (*old_tail).both;
            let result = (*old_tail).value;

            if prev_addr == 0 {
                self.head = null_mut();
                self.tail = null_mut();
            } else {
                self.tail = prev_addr as *mut NodeXOR;
                (*self.tail).both ^= old_tail as usize;
            }

            self.data.pop_back(); // drop the Box<NodeXOR> from VecDeque
            Some(result)
        }
    }

    pub fn get(&self, index: usize) -> Option<i32> {
        let mut prev = null_mut();
        let mut current = self.head;
        let mut i = 0;

        while !current.is_null() && i < index {
            unsafe {
                let next = ((*current).both ^ prev as usize) as *mut NodeXOR;
                prev = current;
                current = next;
            }
            i += 1;
        }

        if current.is_null() {
            None
        } else {
            unsafe { Some((*current).value) }
        }
    }

    pub fn iter(&self) -> ListXORIter<'_> {
        ListXORIter {
            prev: null_mut(),
            current: self.head,
            _marker: std::marker::PhantomData,
        }
    }
}

pub struct ListXORIter<'a> {
    prev: *mut NodeXOR,
    current: *mut NodeXOR,
    _marker: std::marker::PhantomData<&'a NodeXOR>,
}

impl<'a> Iterator for ListXORIter<'a> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            return None;
        }

        unsafe {
            let next = ((*self.current).both ^ self.prev as usize) as *mut NodeXOR;
            let val = (*self.current).value;
            self.prev = self.current;
            self.current = next;
            Some(val)
        }
    }
}

// Daily problem medium 2025 06 06
/*
Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the number of ways it can be decoded.

For example, the message '111' would give 3, since it could be decoded as 'aaa', 'ka', and 'ak'.

You can assume that the messages are decodable. For example, '001' is not allowed.
*/
fn decode_ways_helper_2025_06_06<'a>(message: &'a str, memo: &mut HashMap<&'a str, usize>) -> usize {
    if message.is_empty() {
        return 1;
    }

    if message.starts_with('0') {
        return 0;
    }

    if let Some(&cached) = memo.get(message) {
        return cached;
    }

    let mut count = decode_ways_helper_2025_06_06(&message[1..], memo);

    if message.len() >= 2 {
        let num = &message[..2];
        if let Ok(n) = num.parse::<u8>() {
            if n <= 26 {
                count += decode_ways_helper_2025_06_06(&message[2..], memo);
            }
        }
    }

    memo.insert(message, count);
    count
}

pub fn decode_ways_2025_06_06(message: &str) -> usize {
    let mut ways = HashMap::new();
    decode_ways_helper_2025_06_06(message, &mut ways)
}

// Daily Problem Easy 2025/06/07
/*
A unival tree (which stands for "universal value") is a tree where all nodes under it have the same value.

Given the root to a binary tree, count the number of unival subtrees.

For example, the following tree has 5 unival subtrees:

   0
  / \
 1   0
    / \
   1   0
  / \
 1   1

*/

fn count_unival_helper_daily_2025_06_07(node: &TreeNode<i32>) -> (i32, usize) {
    let (value, mut count) = (node.value, 0);
    let mut node_is_unival = true;

    if let Some(left) = &node.left {
        let (left_univalue, left_count) = count_unival_helper_daily_2025_06_07(&left);
        count += left_count;
        node_is_unival &= value == left_univalue;
    }
    if let Some(right) = &node.right {
        let (right_univalue, right_count) = count_unival_helper_daily_2025_06_07(&right);
        count += right_count;
        node_is_unival &= value == right_univalue;
    }

    if node_is_unival {
        count += 1;
    }

    (value, count)
}

pub fn count_unival_daily_2025_06_07(root: &TreeNode<i32>) -> usize {
    let (_, count) = count_unival_helper_daily_2025_06_07(root);
    count
}

// Daily Hard 2025/06/08
/*
Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0 or negative.

For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5, 1, 1, 5] should return 10, since we pick 5 and 5.

Follow-up: Can you do this in O(N) time and constant space?
*/
pub fn largest_non_adjacent_sum_daily_2025_06_08(numbers: &[i32]) -> i32 {
    if numbers.is_empty() {
        return 0;
    }

    let (mut sum_2_before, mut sum_1_before) = (0, 0);
    for (i, number) in numbers.iter().enumerate() {
        if number <= &0 {
            sum_2_before = sum_1_before.max(sum_2_before);
            sum_1_before = sum_2_before;
            continue;
        }

        if i == 0 {
            sum_2_before = number.clone();
            continue;
        }
        if i == 1 {
            sum_1_before = number.clone();
            continue;
        }

        let old_2_before = sum_2_before;
        sum_2_before = sum_1_before.max(sum_2_before);
        sum_1_before = sum_1_before.max(old_2_before + number);
    }

    sum_2_before.max(sum_1_before)
}

// Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

// TODO

// Meta phone screen 2025/06/03

// Given a list and a window size, return a list of the median value (sort & middle) of each window.
// If 2 middle value, use its average
pub fn median_list_meta_phone_2025_06_03(input: &[i32], window_size: usize) -> Vec<i32> {
    // Assume input size >= window size > 0
    let input_size = input.len();
    let mut result = Vec::new();
    let middle_window_index = window_size / 2;

    for start_index in 0..=(input_size-window_size) {
        let mut window: Vec<i32> = input[start_index..start_index + window_size].to_vec();
        window.sort();

        if window_size % 2 == 0 {
            let average = (window[middle_window_index - 1] + window[middle_window_index]) / 2;
            result.push(average);
        } else {
            result.push(window[middle_window_index]);
        }
    }

    result
}

// Given a Linked List that has value, next, and other, where other can point to any other element of the list, make a deep copy.
pub struct LinkedListWOther {
    value: i32,
    next: Option<Rc<RefCell<LinkedListWOther>>>,
    other: Option<Rc<RefCell<LinkedListWOther>>>,
}

pub fn deep_copy_meta_phone_2025_06_03(input: &Rc<RefCell<LinkedListWOther>>) -> Rc<RefCell<LinkedListWOther>> {
    let mut old_to_new : HashMap<*const RefCell<LinkedListWOther>, Rc<RefCell<LinkedListWOther>>> = HashMap::new();

    let mut input_runner = Some(Rc::clone(input));
    while let Some(input_node) = input_runner {
        let ptr = Rc::as_ptr(&input_node);
        let new_node = Rc::new(RefCell::new(LinkedListWOther {
            value: input_node.borrow().value,
            next: None,
            other: None,
        }));
        old_to_new.insert(ptr, Rc::clone(&new_node));

        input_runner = input_node.borrow().next.clone();
    }

    let mut input_runner = Some(Rc::clone(input));
    while let Some(input_node) = input_runner {
        let new_node = old_to_new.get(&Rc::as_ptr(&input_node)).unwrap();
        if let Some(old_next) = &input_node.borrow().next {
            let new_next = old_to_new.get(&Rc::as_ptr(old_next)).unwrap();
            new_node.borrow_mut().next = Some(new_next.clone());
        }

        if let Some(old_other) = &input_node.borrow().other {
            let new_other = old_to_new.get(&Rc::as_ptr(old_other)).unwrap();
            new_node.borrow_mut().other = Some(new_other.clone());
        }

        input_runner = input_node.borrow().next.clone();
    }

    old_to_new[&Rc::as_ptr(input)].clone()
}

/* We are creating a robot that moves from space to space in a grid printed on the floor.

The room can be thought of as a rectangular map of cells. Each cell can have a single character printed in it. You will be given a two dimensional array of characters representing the room.

Example:

C/Java-ish:

{
  {' ', ' ', ' ', ' ', ' ', 'v'},
  {' ', '>', ' ', '@', ' ', ' '},
  {' ', '^', ' ', ' ', ' ', '<'}
}

// Infinite: no need to detect, but robot shall loop forever
{
  {' ', '>', ' ', ' ', ' ', 'v'},
  {' ', ' ', ' ', '@', ' ', ' '},
  {' ', '^', ' ', ' ', ' ', '<'}
}

// Out of Bound: panic!
{        *
  {' ', ' ', ' ', ' ', ' ', 'v'},
  {' ', ' ', ' ', '@', ' ', ' '},
  {' ', '^', ' ', ' ', ' ', '<'}
}

    The robot starts in the upper-left of the map, facing right.

    Program the robot to move as follows:

        1. Check halt condition: If the robot is standing on the ‘@’ character, print the coordinates of the current character and exit.

        2. If standing on an “arrow” (‘>’, ‘^’, ‘<’, ‘v’), rotate to face the direction indicated.

        3. Move forward to the next space.

        4. Goto 1.

    Program the robot so it will traverse map1 successfully.



    Arithmetic: Subtraction.

    Update movement rules. Before moving, check for these additional characters:

        If the robot lands on a space with a digit printed on it, push the integer represented by the digit onto a stack.

        If the robot lands on a space with a ‘-‘ sign, subtract the top two values on the stack and push the result. (stack subtraction order: a = pop(); b = pop(); push b-a;)

    Update halt condition: If there is a value on top of the stack, return it to the user.


C/Java-ish:

{
  { '8', ' ', ' ', ' ', ' ', ' ', ' ', 'v', ' ' },
  { ' ', '>', ' ', ' ', '@', ' ', ' ', '5', ' ' },
  { ' ', '^', ' ', '-', '1', ' ', '-', '<', ' ' }
}


*/

pub enum Direction {
    Up,
    Down,
    Right,
    Left,
}

pub fn jed_unleash_robot(room: &Vec<Vec<char>>
) -> Result<(usize, usize), String> {
    use Direction::*;

    if room.is_empty() {
        return Err("Empty room".to_string());
    }

    let rows = room.len();
    let cols = room[0].len();

    let mut row = 0;
    let mut col = 0;
    let mut dir = Right;

    loop {
        if row >= rows || col >= cols {
            return Err(format!("Robot exited room @ ({}, {})",
                row, col).to_string());
        }

        match room[row][col] {
            '@' => break,
            '^' => dir = Up,
            '>' => dir = Right,
            '<' => dir = Left,
            'v' => dir = Down,
            _ => {}
        }

        match dir {
            Up => row -= 1,
            Down => row += 1,
            Left => col -= 1,
            Right => col += 1,
        }
    }

    Ok((row, col))
}

pub fn jed_unleash_robot_substract(room: &Vec<Vec<char>>
) -> Result<Option<i32>, String> {
    use Direction::*;

    if room.is_empty() {
        return Err("Empty room".to_string());
    }

    let rows = room.len();
    let cols = room[0].len();

    let mut row = 0;
    let mut col = 0;
    let mut dir = Right;
    let mut stack = Vec::new();

    loop {
        let c = room[row][col];
        if row >= rows || col >= cols {
            return Err(format!("Robot exited room @ ({}, {})",
                row, col).to_string());
        }

        match room[row][col] {
            '@' => break,
            '^' => dir = Up,
            '>' => dir = Right,
            '<' => dir = Left,
            'v' => dir = Down,
            '-' => {
                let err = Err("Cannot substract!");
                match (stack.pop(), stack.pop()) {
                    (None, None) => return err?,
                    (None, _) => return err?,
                    (_, None) => return err?,
                    (Some(a), Some(b)) => stack.push(b - a),
                }
            },
            _ => {}
        }

        if c >= '0' && c <= '9' {
            stack.push((c as i32) - ('0' as i32));
        }

        match dir {
            Up => row -= 1,
            Down => row += 1,
            Left => col -= 1,
            Right => col += 1,
        }
    }

    Ok(stack.pop())
}

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
        self.cat_stack.pop().map(|_| Animal::Cat)
    }

    pub fn dequeue_dog(&mut self) -> Option<Animal> {
        self.dog_stack.pop().map(|_| Animal::Dog)
    }
}

// Problem 4.1: Route Between Nodes
// Given a directed graph, design an algorithm to find out whether there is a route between two nodes.

#[derive(Default)]
pub struct Graph {
    adjacency_list: HashMap<i32, Vec<i32>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    pub fn add_edge(&mut self, from: i32, to: i32) {
        self.adjacency_list.entry(from).or_default().push(to);
    }

    pub fn has_route_dfs(&self, from: i32, to: i32) -> bool {
        let mut visited: HashSet<i32> = HashSet::new();
        self.dfs(from, to, &mut visited)
    }

    pub fn dfs(&self, from: i32, to: i32, visited: &mut HashSet<i32>) -> bool {
        if from == to {
            return true;
        }

        if !visited.insert(from) {
            return false;
        }

        self.adjacency_list
            .get(&from)
            .into_iter()
            .flatten()
            .any(|&neighbor| self.dfs(neighbor, to, visited))
    }

    pub fn has_route_bfs(&self, from: i32, to: i32) -> bool {
        let mut visited: HashSet<i32> = HashSet::new();
        let mut to_visit: VecDeque<i32> = VecDeque::new();

        visited.insert(from);
        to_visit.push_back(from);

        while let Some(current) = to_visit.pop_front() {
            if current == to {
                return true;
            }

            if let Some(neighbors) = self.adjacency_list.get(&current) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        to_visit.push_back(neighbor);
                    }
                }
            }
        }

        false
    }
}

// Problem 4.2: Minimal Tree
// Given a sorted (increasing order) array with unique integer elements, write an algorithm to create a binary search tree with minimal height.

#[derive(Debug, Clone)]
pub struct TreeNode<T> {
    pub value: T,
    pub left: Option<Box<TreeNode<T>>>,
    pub right: Option<Box<TreeNode<T>>>,
}

impl<T> TreeNode<T> {
    pub fn new(value: T) -> Self {
        TreeNode {
            value,
            left: None,
            right: None,
        }
    }

    pub fn with_children(value: T, left: Option<TreeNode<T>>, right: Option<TreeNode<T>>) -> Self {
        TreeNode {
            value,
            left: left.map(Box::new),
            right: right.map(Box::new),
        }
    }
}

pub fn minimal_tree_4_2(arr: &[i32]) -> Option<Box<TreeNode<i32>>> {
    if arr.is_empty() {
        return None;
    }

    let mid = arr.len() / 2;
    let node = TreeNode::new(arr[mid]);
    let left = minimal_tree_4_2(&arr[..mid]);
    let right = minimal_tree_4_2(&arr[mid + 1..]);

    Some(Box::new(TreeNode { value: node.value, left, right }))
}


// Problem 4.3: List of Depths
// Given a binary tree, implement a method to create a linked list of all the nodes at each depth (e.g., if you have a tree with depth D, you'll have D linked lists).

pub fn list_of_depths_dfs_4_3_helper(node: &Option<Box<TreeNode<i32>>>, depth: usize, depths: &mut Vec<LinkedList<i32>>) {
    if let Some(node) = node {
        if depth == depths.len() {
            depths.push(LinkedList::new());
        }

        depths[depth].push_back(node.value);
        list_of_depths_dfs_4_3_helper(&node.left, depth + 1, depths);
        list_of_depths_dfs_4_3_helper(&node.right, depth + 1, depths);
    }
}

pub fn list_of_depths_dfs_4_3(binary_tree: &TreeNode<i32>) -> Vec<LinkedList<i32>> {
    let mut depths = Vec::new();
    list_of_depths_dfs_4_3_helper(&Some(Box::new(binary_tree.clone())), 0, &mut depths);
    depths
}

pub fn list_of_depths_bfs_4_3(binary_tree: &TreeNode<i32>) -> Vec<LinkedList<i32>> {
    let mut depths = Vec::new();
    let mut to_visit = LinkedList::new();
    let mut marked = HashSet::new();

    struct NodeAndDepth<'a> {
        node: &'a TreeNode<i32>,
        depth: usize,
    }

    to_visit.push_back(NodeAndDepth{ node: binary_tree, depth: 0 });
    while let Some(NodeAndDepth { node, depth }) = to_visit.pop_front() {
        if depth == depths.len() {
            depths.push(LinkedList::new());
        }
        depths[depth].push_back(node.value);
        marked.insert(node.value);
        for neighbor in [&node.left, &node.right]
            .into_iter()
            .filter_map(|child| child.as_deref()) {
            if !marked.contains(&neighbor.value) {
                to_visit.push_back(NodeAndDepth { node: neighbor, depth: depth + 1 });
            }
        }
    }
    depths
}

// Problem 4.4: Check Balanced
// Implement a function to check if a binary tree is balanced.
// For the purposes of this question, a balanced tree is defined to be a tree such that no two leaf nodes differ in distance from the root by more than one.

fn check_balanced_leaf_depth_4_4_helper(binary_tree: &TreeNode<i32>, min_leaf_depth: &mut isize, max_leaf_depth: &mut isize, depth: isize) {
    match (&binary_tree.left, &binary_tree.right) {
        (None, None) => {
            // We found a leaf
            if *min_leaf_depth < 0 || depth < *min_leaf_depth {
                *min_leaf_depth = depth;
            }
            if *max_leaf_depth < 0  || depth > *max_leaf_depth {
                *max_leaf_depth = depth;
            }
        },
        (Some(node), None) => {
            check_balanced_leaf_depth_4_4_helper(&node, min_leaf_depth, max_leaf_depth, depth + 1);
        },
        (None, Some(node)) => {
            check_balanced_leaf_depth_4_4_helper(&node, min_leaf_depth, max_leaf_depth, depth + 1);
        },
        (Some(left), Some(right)) => {
            check_balanced_leaf_depth_4_4_helper(&left, min_leaf_depth, max_leaf_depth, depth + 1);
            check_balanced_leaf_depth_4_4_helper(&right, min_leaf_depth, max_leaf_depth, depth + 1);
        }
    }
}

pub fn check_balanced_leaf_depth_4_4(binary_tree: &TreeNode<i32>) -> bool {
    let (mut min_leaf_depth, mut max_leaf_depth) = (-1, -1);
    check_balanced_leaf_depth_4_4_helper(binary_tree, &mut min_leaf_depth, &mut max_leaf_depth, 0);
    println!("{min_leaf_depth} {max_leaf_depth}");
    max_leaf_depth - min_leaf_depth < 2
}

// Now with real balanced definition
fn check_balanced_4_4_helper(binary_tree: &TreeNode<i32>, min_none_depth: &mut isize, max_none_depth: &mut isize, depth: isize) {
    let update_depth = |depth: isize, min_none_depth: &mut isize, max_none_depth: &mut isize| {
        if *min_none_depth < 0 || depth < *min_none_depth {
            *min_none_depth = depth;
        }
        if *max_none_depth < 0  || depth > *max_none_depth {
            *max_none_depth = depth;
        }
    };
    match (&binary_tree.left, &binary_tree.right) {
        (None, None) => {
            update_depth(depth, min_none_depth, max_none_depth);
        },
        (Some(node), None) => {
            update_depth(depth, min_none_depth, max_none_depth);
            check_balanced_4_4_helper(&node, min_none_depth, max_none_depth, depth + 1);
        },
        (None, Some(node)) => {
            update_depth(depth, min_none_depth, max_none_depth);
            check_balanced_4_4_helper(&node, min_none_depth, max_none_depth, depth + 1);
        },
        (Some(left), Some(right)) => {
            check_balanced_4_4_helper(&left, min_none_depth, max_none_depth, depth + 1);
            check_balanced_4_4_helper(&right, min_none_depth, max_none_depth, depth + 1);
        }
    }
}

pub fn check_balanced_4_4(binary_tree: &TreeNode<i32>) -> bool {
    let (mut min_leaf_depth, mut max_leaf_depth) = (-1, -1);
    check_balanced_4_4_helper(binary_tree, &mut min_leaf_depth, &mut max_leaf_depth, 0);
    println!("{min_leaf_depth} {max_leaf_depth}");
    max_leaf_depth - min_leaf_depth < 2
}

// Problem 4.5: Validate BST
// Implement a function to check if a binary tree is a binary search tree.

pub fn validate_bst_4_5_helper(node: &Option<Box<TreeNode<i32>>>) -> (bool, i32 /* min */, i32 /* max */) {
    match node {
        Some(node) => {
            let (left_result, left_min, left_max) = validate_bst_4_5_helper(&node.left);
            let (right_result, right_min, right_max) = validate_bst_4_5_helper(&node.right);
            // Let's not allow duplicates
            if !left_result || !right_result || left_max >= node.value || right_min <= node.value {
                return (false, 0, 0)
            }

            let min = node.value.min(left_min);
            let max = node.value.max(right_max);
            (true, min, max)
        },
        None => (true, i32::MAX, i32::MIN),
    }
}

pub fn validate_bst_4_5(binary_tree: &TreeNode<i32>) -> bool {
    validate_bst_4_5_helper(&Some(Box::new(binary_tree.clone()))).0
}

// Problem 4.6: Successor
// Write an algorithm to find the "next" node (i.e., in-order successor) of a given node in a binary search tree. You may assume that each node has a link to its parent.

type TreeLink = Option<Rc<RefCell<LinkedTreeNode>>>;

#[allow(dead_code)]
pub struct LinkedTreeNode {
    value: i32,
    parent: Option<Weak<RefCell<LinkedTreeNode>>>,
    left: TreeLink,
    right: TreeLink,
}

impl LinkedTreeNode {
     pub fn new(value: i32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(LinkedTreeNode {
            value,
            parent: None,
            left: None,
            right: None,
        }))
    }

    pub fn add_left(parent: &Rc<RefCell<Self>>, value: i32) {
        let child = Rc::new(RefCell::new(LinkedTreeNode {
            value,
            parent: Some(Rc::downgrade(parent)),
            left: None,
            right: None,
        }));
        parent.borrow_mut().left = Some(child);
    }

    pub fn add_right(parent: &Rc<RefCell<Self>>, value: i32) {
        let child = Rc::new(RefCell::new(LinkedTreeNode {
            value,
            parent: Some(Rc::downgrade(parent)),
            left: None,
            right: None,
        }));
        parent.borrow_mut().right = Some(child);
    }
}

pub fn get_upmost_left_child(mut node: Rc<RefCell<LinkedTreeNode>>) -> Rc<RefCell<LinkedTreeNode>> {
    loop {
        let left = node.borrow().left.clone();
        match left {
            Some(next) => node = next,
            None => return node,
        }
    }
}

pub fn find_next_node_4_6(node: Rc<RefCell<LinkedTreeNode>>) -> Option<Rc<RefCell<LinkedTreeNode>>> {
    // Case 1: Right child exists — go down to leftmost node of the right subtree
    if let Some(right) = node.borrow().right.clone() {
        return Some(get_upmost_left_child(right));
    }

    // Case 2: No right child — walk up until we find a node that is a left child
    let mut current = node;

    loop {
        let parent_weak = {
            let current_ref = current.borrow();
            current_ref.parent.clone()
        };

        match parent_weak {
            Some(weak) => {
                if let Some(parent_rc) = weak.upgrade() {
                    let is_left_child = {
                        let parent_ref = parent_rc.borrow();
                        if let Some(left_child) = &parent_ref.left {
                            Rc::ptr_eq(left_child, &current)
                        } else {
                            false
                        }
                    };

                    if is_left_child {
                        return Some(parent_rc);
                    }

                    current = parent_rc;
                } else {
                    break;
                }
            }
            None => break,
        }
    }

    None
}

// Problem 4.7: Build Order
// You are given a list of projects and a list of dependencies
// (which is a list of pairs of projects, where the second project is dependent on the first project).
// All of a project's dependencies must be built before the project is.
// Find a build order that will allow the projects to be built.
// If there is no valid build order, return an error.
// Example:
// Input:
//    projects: a, b, c, d, e, f
//    dependencies: (a, d), (f, b), (b, d), (f, a), (d, c)
// Output: f, e, a, b, d, c

#[derive(Debug)]
pub struct Projects {
    projects: Vec<i32>,
    dependencies: Vec<[i32; 2]>,
}

pub fn build_order_4_7(projects: &Projects) -> Result<Vec<i32>, String> {
    // Translate to Graph
    let mut graph: Graph = Graph::new();
    for dependency in &projects.dependencies {
        graph.add_edge(dependency[0], dependency[1]);
    }

    // In case there are projects without an dependency
    let mut in_degrees = HashMap::new();
    for project in &projects.projects {
        if !graph.adjacency_list.contains_key(&project) {
            graph.adjacency_list.insert(project.clone(), Vec::new());
        }
        in_degrees.insert(*project, 0);
    }

    // Step one: count dependencies for each project
    for dependencies in graph.adjacency_list.values() {
        for dependency in dependencies {
            in_degrees.entry(*dependency).and_modify(|value| *value += 1);
        }
    }

    // Step two: while there are projects at 0, remove them and decrease their dependencies
    let mut ordered_projects = Vec::new();
    while in_degrees.len() > 0 {
        let mut projects_to_be_removed = Vec::new();
        let mut in_degrees_to_decrease = Vec::new();
        for (project, in_degree) in &in_degrees {
            if *in_degree == 0 {
                projects_to_be_removed.push(*project);
                ordered_projects.push(*project);

                for dependency in graph.adjacency_list.get(&project).unwrap() {
                    in_degrees_to_decrease.push(dependency);
                }
            }
        }

        if projects_to_be_removed.is_empty() {
            return Err("Dependency cycle was detected!".to_string());
        }

        for in_degree_to_decrease in in_degrees_to_decrease {
            in_degrees.entry(*in_degree_to_decrease).and_modify(|value| *value -= 1);
        }

        for project in projects_to_be_removed {
            in_degrees.remove(&project);
        }
    }

    Ok(ordered_projects)
}

// Problem 4.8: First Common Ancestor
// You have two very large binary trees: T1, with millions of nodes, and T2, with hundreds of nodes.
// Create an algorithm to determine if T2 is a subtree of T1.
// A tree T2 is a subtree of T1 if there exists a node n in T1 such that the subtree of n is identical to T2.
// That is, if you cut off the tree at node n, the two trees would be identical.

fn first_common_ancestor_4_8_helper_compare(tree_a: &TreeNode<i32>, tree_b: &TreeNode<i32>) -> bool {
    if tree_a.value != tree_b.value {
        return false;
    }

    for (a, b) in [(tree_a.left.as_ref(), tree_b.left.as_ref()), (tree_a.right.as_ref(), tree_b.right.as_ref())] {
        if !match (a, b) {
            (None, None) => true,
            (None, _) => false,
            (_, None) => false,
            (Some(a), Some(b)) => first_common_ancestor_4_8_helper_compare(a.as_ref(), b.as_ref()),
        } {
            return false;
        }
    }

    true
}

pub fn first_common_ancestor_4_8(big_tree: &TreeNode<i32>, small_tree: &TreeNode<i32>) -> bool {
    let mut queue = VecDeque::new();
    queue.push_back(big_tree);

    while let Some(node) = queue.pop_front() {
        if node.value == small_tree.value && first_common_ancestor_4_8_helper_compare(node, small_tree) {
            return true;
        }
        if let Some(left) = node.left.as_ref() {
            queue.push_back(left.as_ref());
        }
        if let Some(right) = node.right.as_ref() {
            queue.push_back(right.as_ref());
        }
    }

    false
}

// Problem 4.9: BST Sequences
// A binary search tree was created by traversing through an array from left to right and inserting each value. Given a binary search tree with distinct elements, print all possible arrays that could have led to this tree.
// Example:
// Input:
//    2
//   / \
//  1   3
// Output:
//    [2, 1, 3], [2, 3, 1]

// TODO

// Problem 4.10: Check Subtree
// T1 and T2 are two very large binary trees, with T1 much bigger than T2. Create an algorithm to check if T2 is a subtree of T1. A tree T2 is a subtree of T1 if there exists a node n in T1 such that the subtree of n is identical to T2. That is, if you cut off the tree at node n, the two trees would be identical.

// TODO

// Problem 4.11: Random Node
// You are implementing a binary tree from scratch which, in addition to insert, find, and delete, has a method getRandomNode() which returns a random node. All nodes should be equally likely to be chosen. Design and implement an algorithm for this. You can use a standard binary tree or binary search tree data structure.
// Extra: You may not use any extra space.

// TODO

// Problem 4.12: Paths with Sum
// You are given a binary tree in which each node contains an integer value (which might be positive or negative). Design an algorithm to count the number of paths that sum to a given value. The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

// TODO

// Problem 5.1: Insertion
// You are given two 32-bit numbers, N and M, and two bit positions i and j.
// Write a method to insert M into N such that M starts at bit j and ends at bit i.
// You can assume that the bits j through i have enough space to fit all of M.
// That is, if M = 10011, you can assume that there are at least 5 bits between j and i.
// Example:
// Input: N = 10000000000, M = 10011, i = 2, j = 6
// Output: N = 10001001100

pub fn insertion_5_1(n: i32, m: i32, i: u8, j: u8) -> i32 {
    let left: u32 = !0 << (j + 1);
    let right: u32 = (1 << i) - 1;
    let m_mask = left | right;

    ((n as u32 & m_mask) | ((m as u32) << i)) as i32
}

// Problem 8.1: Triple Step
// A child is running up a staircase with n steps and can hop either 1 step, 2 steps, or 3 steps at a time.
// Implement a method to count how many possible ways the child can run up the stairs.

pub fn triple_step_8_1(n: usize) -> usize {
    amazon_num_ways_any_steps(n, &vec![1, 2, 3])
}

// Problem 8.2: Robot in a Grid
// Imagine a robot sitting on the upper left corner of a grid with r rows and c columns.
// The robot can only move in two directions: right and down, but certain cells are off-limit.
// Design an algorithm to find a path for the robot from the top left to the bottom right.

#[derive(Debug, PartialEq)]
pub enum RobotGridCell8_2 {
    Inaccessible,
    Accessible,
}

#[derive(Debug, PartialEq)]
pub enum RobotGridMove8_2 {
    Right,
    Down,
}

fn robot_grid_helper_8_2(
    grid: &[Vec<RobotGridCell8_2>],
    row: usize, col: usize,
    path: &mut Vec<RobotGridMove8_2>,
    failed: &mut HashSet<(usize, usize)>
) -> bool {
    use RobotGridCell8_2::*;

    // Helper won't be called on empty grid
    let rows = grid.len();
    let cols = grid[0].len();


    if row >= rows || col >= cols || grid[row][col] == Inaccessible {
        return false;
    }

    if failed.contains(&(row, col)) {
        return false;
    }

    if row == rows - 1 && col == cols - 1 {
        return true;
    }

    if robot_grid_helper_8_2(grid, row, col + 1, path, failed) {
        path.push(RobotGridMove8_2::Right);
        return true;
    }

    if robot_grid_helper_8_2(grid, row + 1, col, path, failed) {
        path.push(RobotGridMove8_2::Down);
        return true;
    }

    failed.insert((row, col));
    false
}

pub fn robot_grid_8_2(grid: &[Vec<RobotGridCell8_2>]) -> Option<Vec<RobotGridMove8_2>> {
    if grid.is_empty() || grid[0].is_empty() {
        return None;
    }

    let mut path = Vec::new();
    let mut failed = HashSet::new();

    if robot_grid_helper_8_2(grid, 0, 0, &mut path, &mut failed) {
        path.reverse();
        Some(path)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use rand::seq::SliceRandom;

    use super::*;

    #[test]
    fn test_sorts() {
        fn assert_sorted(arr: &[i32]) {
            assert!(arr.windows(2).all(|w| w[0] <= w[1]));
        }

        fn test_sort(sort: fn(&mut [i32])) {
            let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
            sort(&mut arr);
            assert_sorted(&arr);

            let mut arr = vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
            sort(&mut arr);
            assert_sorted(&arr);

            let mut arr = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            sort(&mut arr);
            assert_sorted(&arr);

            let mut arr = vec![1];
            sort(&mut arr);
            assert_sorted(&arr);

            let mut arr: Vec<i32> = vec![];
            sort(&mut arr);
            assert_sorted(&arr);

            let mut arr = vec![5, 5, 5, 5, 5];
            sort(&mut arr);
            assert_sorted(&arr);

            let mut arr: Vec<i32> = (1..=1000).collect();
            for _ in 0..10 {
                arr.shuffle(&mut rand::rng());
                sort(&mut arr);
                assert_sorted(&arr);
            }
        }

        test_sort(quicksort);
        test_sort(mergesort_bottomup);
        test_sort(mergesort_topdown);
        test_sort(heapsort);
    }

    #[test]
    fn test_amazon_num_ways() {
        // Base cases
        assert_eq!(amazon_num_ways(0), 1);
        assert_eq!(amazon_num_ways(1), 1);
        assert_eq!(amazon_num_ways(2), 2);

        // Small numbers
        assert_eq!(amazon_num_ways(3), 3); // 1+1+1, 1+2, 2+1
        assert_eq!(amazon_num_ways(4), 5); // 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2

        // Larger numbers
        assert_eq!(amazon_num_ways(5), 8);
        assert_eq!(amazon_num_ways(6), 13);

        // Test a larger value
        assert_eq!(amazon_num_ways(10), 89);

        // Test a reasonably large value (should not panic or overflow)
        let _ = amazon_num_ways(20);
    }

    #[test]
    fn test_amazon_num_ways_any_steps() {
        // Steps: [1, 2]
        let steps = vec![1, 2];
        assert_eq!(amazon_num_ways_any_steps(0, &steps), 0); // n < steps[0]
        assert_eq!(amazon_num_ways_any_steps(1, &steps), 1); // Only one way: 1
        assert_eq!(amazon_num_ways_any_steps(2, &steps), 2); // 1+1, 2
        assert_eq!(amazon_num_ways_any_steps(3, &steps), 3); // 1+1+1, 1+2, 2+1
        assert_eq!(amazon_num_ways_any_steps(4, &steps), 5); // 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2

        // Steps: [2, 3]
        let steps = vec![2, 3];
        assert_eq!(amazon_num_ways_any_steps(1, &steps), 0); // n < steps[0]
        assert_eq!(amazon_num_ways_any_steps(2, &steps), 1); // Only one way: 2
        assert_eq!(amazon_num_ways_any_steps(3, &steps), 1); // Only one way: 3
        assert_eq!(amazon_num_ways_any_steps(4, &steps), 1); // 2+2
        assert_eq!(amazon_num_ways_any_steps(5, &steps), 2); // 2+3, 3+2

        // Steps: [1, 3, 5]
        let steps = vec![1, 3, 5];
        assert_eq!(amazon_num_ways_any_steps(0, &steps), 0);
        assert_eq!(amazon_num_ways_any_steps(1, &steps), 1);
        assert_eq!(amazon_num_ways_any_steps(2, &steps), 1); // 1+1
        assert_eq!(amazon_num_ways_any_steps(3, &steps), 2); // 1+1+1, 3
        assert_eq!(amazon_num_ways_any_steps(4, &steps), 3); // 1+1+1+1, 1+3, 3+1
        assert_eq!(amazon_num_ways_any_steps(5, &steps), 5); // 1+1+1+1+1, 1+1+3, 1+3+1, 3+1+1, 5

        // Steps: [2]
        let steps = vec![2];
        assert_eq!(amazon_num_ways_any_steps(1, &steps), 0);
        assert_eq!(amazon_num_ways_any_steps(2, &steps), 1);
        assert_eq!(amazon_num_ways_any_steps(3, &steps), 0);
        assert_eq!(amazon_num_ways_any_steps(4, &steps), 1); // 2+2

        // Steps: [1]
        let steps = vec![1];
        assert_eq!(amazon_num_ways_any_steps(0, &steps), 0);
        assert_eq!(amazon_num_ways_any_steps(1, &steps), 1);
        assert_eq!(amazon_num_ways_any_steps(2, &steps), 1);
        assert_eq!(amazon_num_ways_any_steps(3, &steps), 1);

        // Steps: [1, 2, 3]
        let steps = vec![1, 2, 3];
        assert_eq!(amazon_num_ways_any_steps(4, &steps), 7); // 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2, 1+3, 3+1

        // Steps: [2, 4]
        let steps = vec![2, 4];
        assert_eq!(amazon_num_ways_any_steps(3, &steps), 0); // can't reach 3
        assert_eq!(amazon_num_ways_any_steps(4, &steps), 2); // 2+2, 4

        // Steps: [5, 10]
        let steps = vec![5, 10];
        assert_eq!(amazon_num_ways_any_steps(4, &steps), 0);
        assert_eq!(amazon_num_ways_any_steps(5, &steps), 1);
        assert_eq!(amazon_num_ways_any_steps(10, &steps), 2); // 5+5, 10

        // Steps: [1, 2], n = 20 (stress test)
        let steps = vec![1, 2];
        let _ = amazon_num_ways_any_steps(20, &steps);

        // Steps: [1, 2, 3, 4, 5], n = 10 (stress test)
        let steps = vec![1, 2, 3, 4, 5];
        let _ = amazon_num_ways_any_steps(10, &steps);

        // Steps: [3, 5, 7], n = 0 (edge case)
        let steps = vec![3, 5, 7];
        assert_eq!(amazon_num_ways_any_steps(0, &steps), 0);

        // Stress test: large n and steps
        let steps = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let n = 30;
        let _ = amazon_num_ways_any_steps(n, &steps);

        // Stress test: very large n with minimal steps
        let steps = vec![1, 2];
        let n = 35;
        let _ = amazon_num_ways_any_steps(n, &steps);

        // Stress test: large steps, small n
        let steps = (1..=100).collect::<Vec<_>>();
        let n = 10;
        let _ = amazon_num_ways_any_steps(n, &steps);

        // Stress test: large steps and large n
        let steps = (1..=20).collect::<Vec<_>>();
        let n = 25;
        let _ = amazon_num_ways_any_steps(n, &steps);

        // Stress test: n much smaller than smallest step
        let steps = vec![100, 200, 300];
        let n = 50;
        let _ = amazon_num_ways_any_steps(n, &steps);
    }

    #[test]
    fn test_daily_easy_2025_06_01() {
        let result = daily_easy_2025_06_01(&[10, 15, 3, 7], 17);
        assert!(result);
    }

    #[test]
    fn test_daily_hard_2025_06_01() {
        for fun in [daily_hard_2025_06_01, daily_hard_2025_06_01_no_div] {
            // Standard cases
            let list = vec![1, 2, 3, 4, 5];
            let expected = vec![120, 60, 40, 30, 24];
            let result = fun(&list);
            assert_eq!(expected, result);

            let list = vec![3, 2, 1];
            let expected = vec![2, 3, 6];
            let result = fun(&list);
            assert_eq!(expected, result);

            // Corner cases
            // Single element
            let list = vec![42];
            let expected = vec![1];
            let result = fun(&list);
            assert_eq!(expected, result);

            // Two elements
            let list = vec![2, 3];
            let expected = vec![3, 2];
            let result = fun(&list);
            assert_eq!(expected, result);

            // Contains zero (for division version, this will panic or be incorrect, but for no-div version, should be zero everywhere except at the zero's index)
            let list = vec![1, 0, 3, 4];
            // For division version, skip this test (would panic or be incorrect)
            if fun as usize == daily_hard_2025_06_01_no_div as usize {
                let result = fun(&list);
                assert_eq!(result, vec![0, 12, 0, 0]);
            }

            // All ones
            let list = vec![1, 1, 1, 1];
            let expected = vec![1, 1, 1, 1];
            let result = fun(&list);
            assert_eq!(expected, result);

            // Empty list
            let list: Vec<i32> = vec![];
            let expected: Vec<i32> = vec![];
            let result = fun(&list);
            assert_eq!(expected, result);
        }
    }

    #[test]
    fn test_daily_harder_2025_06_01() {
        for fun in [daily_harder_2025_06_01_nk, daily_harder_2025_06_01_heapmin] {
            // 2 lists, each sorted, size 3
            let lists = vec![
                vec![1, 4, 7],
                vec![2, 5, 8],
                vec![3, 6, 9],
            ];
            let result = fun(&lists);
            assert_eq!(result, vec![1,2,3,4,5,6,7,8,9]);

            let lists = vec![vec![0; 0]; 0];
            let result = fun(&lists);
            assert_eq!(result, vec![]);

            let lists = [
                vec![1, 2, 3, 4, 5],
            ];
            let result = fun(&lists);
            assert_eq!(result, vec![1,2,3,4,5]);

            let lists = [
                vec![1, 2, 2],
                vec![2, 3, 4],
                vec![2, 2, 5],
            ];
            let result = fun(&lists);
            assert_eq!(result, vec![1,2,2,2,2,2,3,4,5]);

            let lists = [
                vec![-10, -5, 0],
                vec![-7, -3, 2],
                vec![-8, -2, 1],
            ];
            let result = fun(&lists);
            assert_eq!(result, vec![-10, -8, -7, -5, -3, -2, 0, 1, 2]);
        }
    }

    #[test]
    fn test_serialize_deserialize_daily_med_2025_06_02() {
        // Base case: full tree
        let node = BinaryNode::new(
            "root",
            Some(Box::new(BinaryNode::new(
                "left",
                Some(Box::new(BinaryNode::new("left.left", None, None))),
                None,
            ))),
            Some(Box::new(BinaryNode::new("right", None, None))),
        );

        let serialized = serialize_daily_med_2025_06_02(&node)
            .expect("Serialization failed");
        let deserialized = deserialize_daily_med_2025_06_02(&serialized)
            .expect("Deserialization failed");

        assert_eq!(deserialized.left.as_ref().unwrap().left.as_ref().unwrap().val, "left.left");

        // Leaf-only node
        let node = BinaryNode::new("leaf", None, None);
        let serialized = serialize_daily_med_2025_06_02(&node).unwrap();
        let deserialized = deserialize_daily_med_2025_06_02(&serialized).unwrap();
        assert_eq!(deserialized.val, "leaf");
        assert!(deserialized.left.is_none());
        assert!(deserialized.right.is_none());

        // Only left subtree
        let node = BinaryNode::new(
            "root",
            Some(Box::new(BinaryNode::new("left", None, None))),
            None,
        );
        let serialized = serialize_daily_med_2025_06_02(&node).unwrap();
        let deserialized = deserialize_daily_med_2025_06_02(&serialized).unwrap();
        assert_eq!(deserialized.left.as_ref().unwrap().val, "left");
        assert!(deserialized.right.is_none());

        // Only right subtree
        let node = BinaryNode::new(
            "root",
            None,
            Some(Box::new(BinaryNode::new("right", None, None))),
        );
        let serialized = serialize_daily_med_2025_06_02(&node).unwrap();
        let deserialized = deserialize_daily_med_2025_06_02(&serialized).unwrap();
        assert_eq!(deserialized.right.as_ref().unwrap().val, "right");
        assert!(deserialized.left.is_none());

        // Invalid character in value
        let node = BinaryNode::new("bad|value", None, None);
        assert!(serialize_daily_med_2025_06_02(&node).is_err());

        // Invalid deserialization
        assert!(deserialize_daily_med_2025_06_02("[unclosed|node").is_err());
        assert!(deserialize_daily_med_2025_06_02("just text").is_err());
    }

    #[test]
    fn test_first_missing_daily_hard_2025_06_03() {
        // For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0] should give 3.
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![3, 4, -1, 1]), 2);
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![1, 2, 0]), 3);
        // Edge case: empty input
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![]), 1);

        // Edge case: all negative numbers
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![-3, -2, -1]), 1);

        // Edge case: all positive, consecutive starting from 1
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![1, 2, 3, 4, 5]), 6);

        // Edge case: all positive, not starting from 1
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![2, 3, 4]), 1);

        // Edge case: contains duplicates
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![1, 1, 2, 2]), 3);

        // Edge case: contains zero
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![0, 2, 2, 1]), 3);

        // Edge case: large gap
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![100, 101, 102]), 1);

        // Edge case: single element, not 1
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![5]), 1);

        // Edge case: single element, 1
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![1]), 2);

        // Edge case: input with 1 missing in the middle
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![1, 2, 4, 5]), 3);

        // Edge case: input with 1 missing at the start
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![2, 3, 4, 5]), 1);

        // Edge case: input with 1 missing at the end
        assert_eq!(first_missing_daily_hard_2025_06_03(&mut vec![1, 2, 3, 4]), 5);
    }

    #[test]
    fn test_cons_car_cdr_daily_med_2025_06_04() {
        let pair = cons_daily_med_2025_06_04(10, 20);
        assert_eq!(car_daily_med_2025_06_04(&pair), 10);
        assert_eq!(cdr_daily_med_2025_06_04(&pair), 20);

        let pair = cons_daily_med_2025_06_04("hello", "world");
        assert_eq!(car_daily_med_2025_06_04(&pair), "hello");
        assert_eq!(cdr_daily_med_2025_06_04(&pair), "world");
    }

    #[test]
    fn test_first_non_repeating() {
        let input = "leetcode";
        assert_eq!(first_non_repeating(input), 0);

        let input = "loveleetcode";
        assert_eq!(first_non_repeating(input), 2);

        let input = "aabb";
        assert_eq!(first_non_repeating(input), -1);

        let input = "";
        assert_eq!(first_non_repeating(input), -1);
    }

    #[test]
    fn test_who_can_talk_amazon_screen_2025_06_06() {
        use Terrain::*;
        /*
        +------+------+------+------+------+
        | T    | O    | O    | O    | O    |
        | O    | O    | T    | M    | O    |
        | T    | M    | O    | T    | M    |
        | O    | T    | O    | M    | M    |
        | O    | O    | T    | O    | O    |
        +------+------+------+------+------+
         */
        let topology = vec![
            vec![Tower, Open,     Open,  Open,     Open],
            vec![Open,  Open,     Tower, Mountain, Open],
            vec![Tower, Mountain, Open,  Tower,    Mountain],
            vec![Open,  Tower,    Open,  Mountain, Mountain],
            vec![Open,  Open,     Tower, Open,     Open],
        ];
        let result = who_can_talk_amazon_screen_2025_06_06(&topology);
        let expected = vec![
            (Coordinate { row: 0, col: 0 }, Coordinate { row: 2, col: 0 }),
            (Coordinate { row: 1, col: 2 }, Coordinate { row: 4, col: 2 }),
        ];
        assert_eq!(expected, result);
    }

    #[test]
    fn test_listxor_basic_operations() {
        let mut list = ListXOR::new();
        // Initially empty
        assert!(list.is_empty());

        // Push elements to the front and back
        list.push_front(1);
        assert_eq!(list.front(), Some(1));
        assert_eq!(list.back(), Some(1));

        list.push_back(2);
        assert_eq!(list.front(), Some(1));
        assert_eq!(list.back(), Some(2));

        list.push_front(0);
        assert_eq!(list.front(), Some(0));
        assert_eq!(list.back(), Some(2));

        // Pop from front and back
        assert_eq!(list.pop_front(), Some(0));
        assert_eq!(list.front(), Some(1));

        assert_eq!(list.pop_back(), Some(2));
        assert_eq!(list.back(), Some(1));

        assert_eq!(list.pop_front(), Some(1));
        assert!(list.is_empty());
        assert_eq!(list.pop_back(), None);
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn test_listxor_mixed_operations() {
        let mut list = ListXOR::new();
        list.push_back(10);
        list.push_front(20);
        list.push_back(30);
        list.push_front(40);

        assert_eq!(list.get(3), Some(30));
        assert_eq!(list.front(), Some(40));
        assert_eq!(list.back(), Some(30));
        assert_eq!(list.iter().collect::<Vec<_>>(), vec![40, 20, 10, 30]);
        assert_eq!(list.pop_back(), Some(30));
        assert_eq!(list.pop_front(), Some(40));
        assert_eq!(list.pop_back(), Some(10));
        assert_eq!(list.pop_front(), Some(20));
        assert!(list.is_empty());
    }

    #[test]
    fn test_listxor_single_element() {
        let mut list = ListXOR::new();
        list.push_back(99);
        assert_eq!(list.front(), Some(99));
        assert_eq!(list.back(), Some(99));
        assert_eq!(list.pop_front(), Some(99));
        assert!(list.is_empty());
    }

    #[test]
    fn test_listxor_clear() {
        let mut list = ListXOR::new();
        for i in 0..10 {
            list.push_back(i);
        }
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn test_decode_ways_2025_06_06_basic() {
        // "12" -> "AB", "L"
        assert_eq!(decode_ways_2025_06_06("12"), 2);

        // "226" -> "BBF", "BZ", "VF"
        assert_eq!(decode_ways_2025_06_06("226"), 3);

        // "0" -> invalid
        assert_eq!(decode_ways_2025_06_06("0"), 0);

        // "06" -> invalid
        assert_eq!(decode_ways_2025_06_06("06"), 0);

        // "10" -> "J"
        assert_eq!(decode_ways_2025_06_06("10"), 1);

        // "27" -> "BG"
        assert_eq!(decode_ways_2025_06_06("27"), 1);

        // "101" -> "JA"
        assert_eq!(decode_ways_2025_06_06("101"), 1);

        // "100" -> invalid
        assert_eq!(decode_ways_2025_06_06("100"), 0);

        // "11106" -> "AAJF", "KJF"
        assert_eq!(decode_ways_2025_06_06("11106"), 2);

        // "111" -> "AAA", "KA", "AK"
        assert_eq!(decode_ways_2025_06_06("111"), 3);

        // "1234" -> "ABCD", "LCD", "AWD"
        assert_eq!(decode_ways_2025_06_06("1234"), 3);

        // "2611055971756562" (stress test, should not panic)
        let _ = decode_ways_2025_06_06("2611055971756562");
    }

    #[test]
    fn test_count_unival_daily_2025_06_07() {
        // Example from the problem description:
        //    0
        //   / \
        //  1   0
        //     / \
        //    1   0
        //   / \
        //  1   1
        let tree = TreeNode {
            value: 0,
            left: Some(Box::new(TreeNode {
                value: 1,
                left: None,
                right: None,
            })),
            right: Some(Box::new(TreeNode {
                value: 0,
                left: Some(Box::new(TreeNode {
                    value: 1,
                    left: Some(Box::new(TreeNode::new(1))),
                    right: Some(Box::new(TreeNode::new(1))),
                })),
                right: Some(Box::new(TreeNode::new(0))),
            })),
        };
        assert_eq!(count_unival_daily_2025_06_07(&tree), 5);

        // Single node (should be 1)
        let tree = TreeNode::new(42);
        assert_eq!(count_unival_daily_2025_06_07(&tree), 1);

        // All nodes same value (full unival tree)
        let tree = TreeNode {
            value: 1,
            left: Some(Box::new(TreeNode {
                value: 1,
                left: Some(Box::new(TreeNode::new(1))),
                right: Some(Box::new(TreeNode::new(1))),
            })),
            right: Some(Box::new(TreeNode {
                value: 1,
                left: Some(Box::new(TreeNode::new(1))),
                right: Some(Box::new(TreeNode::new(1))),
            })),
        };
        assert_eq!(count_unival_daily_2025_06_07(&tree), 7);

        // No unival subtrees except leaves
        let tree = TreeNode {
            value: 1,
            left: Some(Box::new(TreeNode::new(2))),
            right: Some(Box::new(TreeNode::new(3))),
        };
        assert_eq!(count_unival_daily_2025_06_07(&tree), 2);

        // Left-skewed tree
        let tree = TreeNode {
            value: 1,
            left: Some(Box::new(TreeNode {
                value: 1,
                left: Some(Box::new(TreeNode::new(1))),
                right: None,
            })),
            right: None,
        };
        assert_eq!(count_unival_daily_2025_06_07(&tree), 3);
    }

    #[test]
    fn test_largest_non_adjacent_sum_daily_2025_06_08() {
        // Example from the problem description
        let numbers = vec![2, 4, 6, 2, 5];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 13);

        let numbers = vec![5, 1, 1, 5];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 10);

        // All negatives
        let numbers = vec![-1, -2, -3, -4];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 0);

        // All zeros
        let numbers = vec![0, 0, 0, 0];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 0);

        // Single element
        let numbers = vec![7];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 7);

        // Two elements
        let numbers = vec![3, 10];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 10);

        // Empty input
        let numbers: Vec<i32> = vec![];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 0);

        // Large input, alternating positives and negatives
        let numbers = vec![10, -1, 20, -2, 30, -3, 40, -4, 50];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 150);

        // Input with zeros and positives
        let numbers = vec![0, 5, 0, 10, 0, 15];
        assert_eq!(largest_non_adjacent_sum_daily_2025_06_08(&numbers), 30);
    }

    #[test]
    fn test_median_list_meta_phone_2025_06_03_basic() {
        // Odd window size
        let input = vec![1, 3, 2, 6, 7, 8, 9];
        let window_size = 3;
        let result = median_list_meta_phone_2025_06_03(&input, window_size);
        // Windows: [1,3,2]=2, [3,2,6]=3, [2,6,7]=6, [6,7,8]=7, [7,8,9]=8
        assert_eq!(result, vec![2, 3, 6, 7, 8]);

        // Even window size
        let input = vec![1, 2, 3, 4, 5, 6];
        let window_size = 4;
        let result = median_list_meta_phone_2025_06_03(&input, window_size);
        // Windows: [1,2,3,4]=2, [2,3,4,5]=3, [3,4,5,6]=4
        assert_eq!(result, vec![2, 3, 4]);
    }

    #[test]
    fn test_median_list_meta_phone_2025_06_03_edge_cases() {
        // Window size 1 (should return the input itself)
        let input = vec![5, 4, 3, 2, 1];
        let window_size = 1;
        let result = median_list_meta_phone_2025_06_03(&input, window_size);
        assert_eq!(result, input);

        // Window size equals input length (should return single median)
        let input = vec![10, 20, 30, 40, 50];
        let window_size = 5;
        let result = median_list_meta_phone_2025_06_03(&input, window_size);
        // Sorted: [10,20,30,40,50] median is 30
        assert_eq!(result, vec![30]);
    }

    #[test]
    fn test_median_list_meta_phone_2025_06_03_duplicates_and_negatives() {
        let input = vec![1, 2, 2, 2, 3, 4, 4, 5, -1, 0];
        let window_size = 5;
        let result = median_list_meta_phone_2025_06_03(&input, window_size);
        // Windows: [1,2,2,2,3]=2, [2,2,2,3,4]=2, [2,2,3,4,4]=3, [2,3,4,4,5]=4, [3,4,4,5,-1]=4, [4,4,5,-1,0]=4
        assert_eq!(result, vec![2, 2, 3, 4, 4, 4]);
    }

    #[test]
    fn test_median_list_meta_phone_2025_06_03_even_window_average() {
        let input = vec![1, 2, 3, 4];
        let window_size = 2;
        let result = median_list_meta_phone_2025_06_03(&input, window_size);
        // Windows: [1,2]=1, [2,3]=2, [3,4]=3 (since (1+2)/2=1, (2+3)/2=2, (3+4)/2=3)
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_deep_copy_meta_phone_2025_06_03() {
        // Helper to build a list of 3 nodes: 1 -> 2 -> 3, with .other pointers
        let node1 = Rc::new(RefCell::new(LinkedListWOther {
            value: 1,
            next: None,
            other: None,
        }));
        let node2 = Rc::new(RefCell::new(LinkedListWOther {
            value: 2,
            next: None,
            other: None,
        }));
        let node3 = Rc::new(RefCell::new(LinkedListWOther {
            value: 3,
            next: None,
            other: None,
        }));

        // Link next pointers
        node1.borrow_mut().next = Some(node2.clone());
        node2.borrow_mut().next = Some(node3.clone());

        // Set .other pointers
        node1.borrow_mut().other = Some(node3.clone()); // 1.other -> 3
        node2.borrow_mut().other = Some(node1.clone()); // 2.other -> 1
        node3.borrow_mut().other = Some(node2.clone()); // 3.other -> 2

        // Deep copy
        let copy_head = deep_copy_meta_phone_2025_06_03(&node1);

        // Check values and structure
        let orig_nodes = [node1, node2, node3];
        let mut orig_runner = Some(orig_nodes[0].clone());
        let mut copy_runner = Some(copy_head.clone());
        let mut i = 0;
        let mut copy_nodes = vec![];

        // Collect copy nodes and check values
        while let (Some(orig), Some(copy)) = (orig_runner, copy_runner) {
            assert_eq!(orig.borrow().value, copy.borrow().value);
            copy_nodes.push(copy.clone());
            orig_runner = orig.borrow().next.clone();
            copy_runner = copy.borrow().next.clone();
            i += 1;
        }
        assert_eq!(i, 3);

        // Check .other pointers point to correct nodes (by value, not by Rc address)
        // and that copy nodes are not the same Rc as original nodes
        for (orig, copy) in orig_nodes.iter().zip(copy_nodes.iter()) {
            assert!(!Rc::ptr_eq(orig, copy));
            let orig_other_val = orig.borrow().other.as_ref().unwrap().borrow().value;
            let copy_other_val = copy.borrow().other.as_ref().unwrap().borrow().value;
            assert_eq!(orig_other_val, copy_other_val);
        }

        // Check .other pointers in the copy point to nodes in the copy, not to original nodes
        for copy in &copy_nodes {
            let copy_other = copy.borrow().other.as_ref().unwrap().clone();
            assert!(copy_nodes.iter().any(|n| Rc::ptr_eq(n, &copy_other)));
        }
    }

    #[test]
    fn test_jed_unleash_robot() {
        let room = vec![
            vec![' ', ' ', ' ', ' ', ' ', 'v'],
            vec![' ', '>', ' ', '@', ' ', ' '],
            vec![' ', '^', ' ', ' ', ' ', '<'],
        ];

        assert_eq!(jed_unleash_robot(&room), Ok((1, 3)));

        let room = vec![
            vec!['8', ' ', ' ', ' ', ' ', ' ', ' ', 'v', ' '],
            vec![' ', '>', ' ', ' ', '@', ' ', ' ', '5', ' '],
            vec![' ', '^', ' ', '-', '1', ' ', '-', '<', ' '],
        ];

        assert_eq!(jed_unleash_robot_substract(&room), Ok(Some(2)));
    }

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

    #[test]
    fn test_graph_has_route() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);

        // Test direct route
        assert_eq!(graph.has_route_dfs(1, 2), true);

        // Test indirect route
        assert_eq!(graph.has_route_dfs(1, 4), true);

        // Test no route
        assert_eq!(graph.has_route_dfs(4, 1), false);

        // Test route to itself
        assert_eq!(graph.has_route_dfs(1, 1), true);

        // Test disconnected nodes
        graph.add_edge(5, 6);
        assert_eq!(graph.has_route_dfs(1, 5), false);
        assert_eq!(graph.has_route_dfs(5, 6), true);

        // Test cyclic graph
        graph.add_edge(4, 1);
        assert_eq!(graph.has_route_dfs(1, 3), true);
        assert_eq!(graph.has_route_dfs(3, 1), true);
    }

    #[test]
    fn test_graph_empty() {
        let graph = Graph::new();

        // Test no route in an empty graph
        assert_eq!(graph.has_route_dfs(1, 2), false);
    }

    #[test]
    fn test_graph_single_node() {
        let mut graph = Graph::new();
        graph.add_edge(1, 1);

        // Test route to itself in a single-node graph
        assert_eq!(graph.has_route_dfs(1, 1), true);

        // Test no route to another node
        assert_eq!(graph.has_route_dfs(1, 2), false);
    }

    #[test]
    fn test_graph_multiple_paths() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);
        graph.add_edge(3, 4);

        // Test multiple paths to the same node
        assert_eq!(graph.has_route_dfs(1, 4), true);
    }

    #[test]
    fn test_graph_no_edges() {
        let mut graph = Graph::new();
        graph.add_edge(1, 1);
        graph.add_edge(2, 2);

        // Test no route between disconnected nodes
        assert_eq!(graph.has_route_dfs(1, 2), false);
    }

    #[test]
    fn test_graph_has_route_bfs() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);

        // Test direct route
        assert_eq!(graph.has_route_bfs(1, 2), true);

        // Test indirect route
        assert_eq!(graph.has_route_bfs(1, 4), true);

        // Test no route
        assert_eq!(graph.has_route_bfs(4, 1), false);

        // Test route to itself
        assert_eq!(graph.has_route_bfs(1, 1), true);

        // Test disconnected nodes
        graph.add_edge(5, 6);
        assert_eq!(graph.has_route_bfs(1, 5), false);
        assert_eq!(graph.has_route_bfs(5, 6), true);

        // Test cyclic graph
        graph.add_edge(4, 1);
        assert_eq!(graph.has_route_bfs(1, 3), true);
        assert_eq!(graph.has_route_bfs(3, 1), true);
    }

    #[test]
    fn test_graph_empty_bfs() {
        let graph = Graph::new();

        // Test no route in an empty graph
        assert_eq!(graph.has_route_bfs(1, 2), false);
    }

    #[test]
    fn test_graph_single_node_bfs() {
        let mut graph = Graph::new();
        graph.add_edge(1, 1);

        // Test route to itself in a single-node graph
        assert_eq!(graph.has_route_bfs(1, 1), true);

        // Test no route to another node
        assert_eq!(graph.has_route_bfs(1, 2), false);
    }

    #[test]
    fn test_graph_multiple_paths_bfs() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);
        graph.add_edge(3, 4);

        // Test multiple paths to the same node
        assert_eq!(graph.has_route_bfs(1, 4), true);
    }

    #[test]
    fn test_graph_no_edges_bfs() {
        let mut graph = Graph::new();
        graph.add_edge(1, 1);
        graph.add_edge(2, 2);

        // Test no route between disconnected nodes
        assert_eq!(graph.has_route_bfs(1, 2), false);
    }

    #[test]
    fn test_minimal_tree_4_2() {
        // Test with an empty array
        let arr: Vec<i32> = vec![];
        let result = minimal_tree_4_2(&arr);
        assert!(result.is_none());

        // Test with a single element
        let arr = vec![10];
        let result = minimal_tree_4_2(&arr);
        assert!(result.is_some());
        let root = result.unwrap();
        assert_eq!(root.value, 10);
        assert!(root.left.is_none());
        assert!(root.right.is_none());

        // Test with two elements
        let arr = vec![10, 20];
        let result = minimal_tree_4_2(&arr);
        assert!(result.is_some());
        let root = result.unwrap();
        assert_eq!(root.value, 20);
        assert!(root.right.is_none());
        assert!(root.left.is_some());
        assert_eq!(root.left.unwrap().value, 10);

        // Test with three elements
        let arr = vec![10, 20, 30];
        let result = minimal_tree_4_2(&arr);
        assert!(result.is_some());
        let root = result.unwrap();
        assert_eq!(root.value, 20);
        assert!(root.left.is_some());
        assert!(root.right.is_some());
        assert_eq!(root.left.unwrap().value, 10);
        assert_eq!(root.right.unwrap().value, 30);

        // Test with multiple elements
        let arr = vec![1, 2, 3, 4, 5, 6, 7];
        let result = minimal_tree_4_2(&arr);
        assert!(result.is_some());
        let root = result.unwrap();
        assert_eq!(root.value, 4);

        let left = root.left.unwrap();
        let right = root.right.unwrap();
        assert_eq!(left.value, 2);
        assert_eq!(right.value, 6);

        let left_left = left.left.unwrap();
        let left_right = left.right.unwrap();
        let right_left = right.left.unwrap();
        let right_right = right.right.unwrap();

        assert_eq!(left_left.value, 1);
        assert_eq!(left_right.value, 3);
        assert_eq!(right_left.value, 5);
        assert_eq!(right_right.value, 7);
    }

    #[test]
    fn test_list_of_depths_4_3() {
        let dfs_bfs = [list_of_depths_dfs_4_3, list_of_depths_bfs_4_3];
        for fun in dfs_bfs {
            // Test with a single-node tree
            let single_node_tree = TreeNode::new(10);
            let result = fun(&single_node_tree);
            let mut expected = LinkedList::new();
            expected.push_back(10);
            assert_eq!(result, vec![expected]);

            // Test with a balanced tree
            let mut root = TreeNode::new(1);
            root.left = Some(Box::new(TreeNode::new(2)));
            root.right = Some(Box::new(TreeNode::new(3)));
            root.left.as_mut().unwrap().left = Some(Box::new(TreeNode::new(4)));
            root.left.as_mut().unwrap().right = Some(Box::new(TreeNode::new(5)));
            root.right.as_mut().unwrap().left = Some(Box::new(TreeNode::new(6)));
            root.right.as_mut().unwrap().right = Some(Box::new(TreeNode::new(7)));

            let result = fun(&root);

            let mut level_1 = LinkedList::new();
            level_1.push_back(1);

            let mut level_2 = LinkedList::new();
            level_2.push_back(2);
            level_2.push_back(3);

            let mut level_3 = LinkedList::new();
            level_3.push_back(4);
            level_3.push_back(5);
            level_3.push_back(6);
            level_3.push_back(7);

            assert_eq!(result, vec![level_1, level_2, level_3]);

            // Test with an unbalanced tree
            let mut unbalanced_root = TreeNode::new(1);
            unbalanced_root.left = Some(Box::new(TreeNode::new(2)));
            unbalanced_root.left.as_mut().unwrap().left = Some(Box::new(TreeNode::new(3)));
            unbalanced_root.left.as_mut().unwrap().left.as_mut().unwrap().left = Some(Box::new(TreeNode::new(4)));

            let result = fun(&unbalanced_root);

            let mut level_1 = LinkedList::new();
            level_1.push_back(1);

            let mut level_2 = LinkedList::new();
            level_2.push_back(2);

            let mut level_3 = LinkedList::new();
            level_3.push_back(3);

            let mut level_4 = LinkedList::new();
            level_4.push_back(4);

            assert_eq!(result, vec![level_1, level_2, level_3, level_4]);
        }
    }

    #[test]
    fn test_check_balanced_4_4() {
        // Single node
        let root = TreeNode::new(1);
        assert!(check_balanced_leaf_depth_4_4(&root));
        assert!(check_balanced_4_4(&root));

        // Perfectly balanced
        let tree = TreeNode::with_children(
            1,
            Some(TreeNode::with_children(2, Some(TreeNode::new(4)), Some(TreeNode::new(5)))),
            Some(TreeNode::with_children(3, Some(TreeNode::new(6)), Some(TreeNode::new(7)))),
        );
        assert!(check_balanced_leaf_depth_4_4(&tree));
        assert!(check_balanced_4_4(&root));

        // Depth differs by 1
        let mut root = TreeNode::new(1);
        root.left = Some(Box::new(TreeNode::new(2)));
        root.right = Some(Box::new(TreeNode::with_children(3, Some(TreeNode::new(4)), Some(TreeNode::new(5)))));
        // Note: if the only leaf was deep in the left, our algo would consider it balanced, as per the instructions
        assert!(check_balanced_leaf_depth_4_4(&root));

        // The real algorithm should be able to detect imbalance outside leaves
        root.left = None;
        assert!(!check_balanced_4_4(&root));

        // Unbalanced
        let mut deep_branch = TreeNode::new(2);
        deep_branch.left = Some(Box::new(TreeNode::new(3)));
        deep_branch.left.as_mut().unwrap().left = Some(Box::new(TreeNode::new(4)));
        let root = TreeNode::with_children(1, Some(deep_branch), Some(TreeNode::new(5)));
        assert!(!check_balanced_leaf_depth_4_4(&root));
        assert!(!check_balanced_4_4(&root));
    }

    #[test]
    fn test_validate_bst_4_5() {
        // Valid BST
        let tree = TreeNode::with_children(4, Some(TreeNode::with_children(2, Some(TreeNode::new(1)), Some(TreeNode::new(3)))), Some(TreeNode::with_children(6, Some(TreeNode::new(5)), Some(TreeNode::new(7)))));
        assert!(validate_bst_4_5(&tree));

        // Not valid from the left
        let tree = TreeNode::with_children(4, Some(TreeNode::with_children(2, Some(TreeNode::new(1)), Some(TreeNode::new(5)))), Some(TreeNode::with_children(6, Some(TreeNode::new(5)), Some(TreeNode::new(7)))));
        assert!(!validate_bst_4_5(&tree));

        // Not valid from the right
        let tree = TreeNode::with_children(4, Some(TreeNode::with_children(2, Some(TreeNode::new(1)), Some(TreeNode::new(3)))), Some(TreeNode::with_children(6, Some(TreeNode::new(1)), Some(TreeNode::new(7)))));
        assert!(!validate_bst_4_5(&tree));

        // Duplicates
        let tree = TreeNode::with_children(4, Some(TreeNode::with_children(2, Some(TreeNode::new(1)), Some(TreeNode::new(3)))), Some(TreeNode::with_children(6, Some(TreeNode::new(4)), Some(TreeNode::new(7)))));
        assert!(!validate_bst_4_5(&tree));

        // Single-node tree
        let tree = TreeNode::new(42);
        assert!(validate_bst_4_5(&tree));

        // Right-skewed invalid tre
        let tree = TreeNode::with_children(3, None, Some(TreeNode::with_children(2, None, Some(TreeNode::new(1)))));
        assert!(!validate_bst_4_5(&tree));

        // Valid unbalanced but BST
        let tree = TreeNode::with_children(5, Some(TreeNode::with_children(3, Some(TreeNode::with_children(2, Some(TreeNode::new(1)), None)), None)), None);
        assert!(validate_bst_4_5(&tree));
     }

    #[test]
    fn test_find_next_node_4_6() {
        // Build the following BST with parent pointers:
        //         20
        //        /  \
        //      10    30
        //     /  \     \
        //    5   15     35
        //         \    /
        //         17  32
        //        /     \
        //      16       33

        // Build tree bottom-up
        let root = LinkedTreeNode::new(20);

        // Left subtree
        LinkedTreeNode::add_left(&root, 10);
        let left = root.borrow().left.as_ref().unwrap().clone();
        LinkedTreeNode::add_left(&left, 5);
        LinkedTreeNode::add_right(&left, 15);
        let left_right = left.borrow().right.as_ref().unwrap().clone();
        LinkedTreeNode::add_right(&left_right, 17);
        let left_right_right = left_right.borrow().right.as_ref().unwrap().clone();
        LinkedTreeNode::add_left(&left_right_right, 16);

        // Right subtree
        LinkedTreeNode::add_right(&root, 30);
        let right = root.borrow().right.as_ref().unwrap().clone();
        LinkedTreeNode::add_right(&right, 35);
        let right_right = right.borrow().right.as_ref().unwrap().clone();
        LinkedTreeNode::add_left(&right_right, 32);
        let right_right_left = right_right.borrow().left.as_ref().unwrap().clone();
        LinkedTreeNode::add_right(&right_right_left, 33);

        // Helper to find node by value (DFS)
        fn find_node(node: &TreeLink, value: i32) -> Option<Rc<RefCell<LinkedTreeNode>>> {
            if let Some(n) = node {
                if n.borrow().value == value {
                    return Some(n.clone());
                }
                if let Some(found) = find_node(&n.borrow().left, value) {
                    return Some(found);
                }
                if let Some(found) = find_node(&n.borrow().right, value) {
                    return Some(found);
                }
            }
            None
        }

        // 1. Node with right child: 10 -> 15 (leftmost of right subtree)
        let node_10 = find_node(&Some(root.clone()), 10).unwrap();
        assert!(find_next_node_4_6(node_10.clone()).map(|n| n.borrow().value) == Some(15));

        // 2. Node with right child and right subtree has left descendants: 15 -> 16
        let node_15 = find_node(&Some(root.clone()), 15).unwrap();
        assert!(find_next_node_4_6(node_15.clone()).map(|n| n.borrow().value) == Some(16));

        // 3. Node with no right child, is left child: 5 -> 10
        let node_5 = find_node(&Some(root.clone()), 5).unwrap();
        assert!(find_next_node_4_6(node_5.clone()).map(|n| n.borrow().value) == Some(10));

        // 4. Node with no right child, is right child: 16 -> 17
        let node_16 = find_node(&Some(root.clone()), 16).unwrap();
        assert!(find_next_node_4_6(node_16.clone()).map(|n| n.borrow().value) == Some(17));

        // 5. Node with no right child, must go up multiple parents: 17 -> 20
        let node_17 = find_node(&Some(root.clone()), 17).unwrap();
        assert!(find_next_node_4_6(node_17.clone()).map(|n| n.borrow().value) == Some(20));

        // 6. Node with right child: 30 -> 32
        let node_30 = find_node(&Some(root.clone()), 30).unwrap();
        assert!(find_next_node_4_6(node_30.clone()).map(|n| n.borrow().value) == Some(32));

        // 7. Node with no right child, is rightmost: 35 -> None
        let node_35 = find_node(&Some(root.clone()), 35).unwrap();
        assert!(find_next_node_4_6(node_35.clone()).is_none());

        // 8. Root node: 20 -> 30
        assert!(find_next_node_4_6(root.clone()).map(|n| n.borrow().value) == Some(30));

        // 9. Node that is a right child that needs to go up twice: 33 -> 35
        let node_33 = find_node(&Some(root.clone()), 33).unwrap();
        assert!(find_next_node_4_6(node_33.clone()).map(|n| n.borrow().value) == Some(35));
    }

    #[test]
    fn test_build_order_4_7_basic() {
        // Example from the problem statement
        let projects = Projects {
            projects: vec![1, 2, 3, 4, 5, 6],
            dependencies: vec![[1, 4], [6, 2], [2, 4], [6, 1], [4, 3]],
        };
        let result = build_order_4_7(&projects);
        assert!(result.is_ok());
        let order = result.unwrap();
        // The only requirement is that dependencies come before dependents
        let pos = |x| order.iter().position(|&y| y == x).unwrap();
        assert!(pos(6) < pos(2));
        assert!(pos(6) < pos(1));
        assert!(pos(1) < pos(4));
        assert!(pos(2) < pos(4));
        assert!(pos(4) < pos(3));
        // All projects included
        for p in &projects.projects {
            assert!(order.contains(p));
        }
    }

    #[test]
    fn test_build_order_4_7_no_dependencies() {
        let projects = Projects {
            projects: vec![1, 2, 3],
            dependencies: vec![],
        };
        let result = build_order_4_7(&projects);
        assert!(result.is_ok());
        let order = result.unwrap();
        // Any order is valid
        assert_eq!(order.len(), 3);
        for p in &projects.projects {
            assert!(order.contains(p));
        }
    }

    #[test]
    fn test_build_order_4_7_single_project() {
        let projects = Projects {
            projects: vec![42],
            dependencies: vec![],
        };
        let result = build_order_4_7(&projects);
        assert_eq!(result, Ok(vec![42]));
    }

    #[test]
    fn test_build_order_4_7_linear_dependencies() {
        // 1 -> 2 -> 3 -> 4
        let projects = Projects {
            projects: vec![1, 2, 3, 4],
            dependencies: vec![[1, 2], [2, 3], [3, 4]],
        };
        let result = build_order_4_7(&projects);
        assert!(result.is_ok());
        let order = result.unwrap();
        let pos = |x| order.iter().position(|&y| y == x).unwrap();
        assert!(pos(1) < pos(2));
        assert!(pos(2) < pos(3));
        assert!(pos(3) < pos(4));
    }

    #[test]
    fn test_build_order_4_7_multiple_roots() {
        // 1 -> 3, 2 -> 3, 4 (no dependencies)
        let projects = Projects {
            projects: vec![1, 2, 3, 4],
            dependencies: vec![[1, 3], [2, 3]],
        };
        let result = build_order_4_7(&projects);
        assert!(result.is_ok());
        let order = result.unwrap();
        let pos = |x| order.iter().position(|&y| y == x).unwrap();
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
        // 4 can be anywhere
        assert!(order.contains(&4));
    }

    #[test]
    fn test_build_order_4_7_cycle() {
        // 1 -> 2 -> 3 -> 1 (cycle)
        let projects = Projects {
            projects: vec![1, 2, 3],
            dependencies: vec![[1, 2], [2, 3], [3, 1]],
        };
        let result = build_order_4_7(&projects);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Dependency cycle was detected!".to_string());
    }

    #[test]
    fn test_build_order_4_7_disconnected_graph() {
        // 1 -> 2, 3 (no dependencies), 4 -> 5
        let projects = Projects {
            projects: vec![1, 2, 3, 4, 5],
            dependencies: vec![[1, 2], [4, 5]],
        };
        let result = build_order_4_7(&projects);
        assert!(result.is_ok());
        let order = result.unwrap();
        let pos = |x| order.iter().position(|&y| y == x).unwrap();
        assert!(pos(1) < pos(2));
        assert!(pos(4) < pos(5));
        assert!(order.contains(&3));
    }

    #[test]
    fn test_build_order_4_7_empty_projects() {
        let projects = Projects {
            projects: vec![],
            dependencies: vec![],
        };
        let result = build_order_4_7(&projects);
        assert_eq!(result, Ok(vec![]));
    }

    #[test]
    fn test_first_common_ancestor_4_8() {
        // Helper to build a tree from nested tuples: (value, left, right)
        fn build_tree(data: Option<(i32, Option<Box<TreeNode<i32>>>, Option<Box<TreeNode<i32>>>)>) -> Option<Box<TreeNode<i32>>> {
            data.map(|(v, l, r)| Box::new(TreeNode { value: v, left: l, right: r }))
        }

        // Tree T1:
        //        10
        //      /    \
        //     5      15
        //    / \    /  \
        //   3   7  12  18
        //      /
        //     6
        let t1 = TreeNode {
            value: 10,
            left: build_tree(Some((
                5,
                build_tree(Some((3, None, None))),
                build_tree(Some((
                    7,
                    build_tree(Some((6, None, None))),
                    None,
                ))),
            ))),
            right: build_tree(Some((
                15,
                build_tree(Some((12, None, None))),
                build_tree(Some((18, None, None))),
            ))),
        };

        // T2 is a subtree of T1 (matches the left subtree rooted at 5)
        let t2 = TreeNode {
            value: 5,
            left: build_tree(Some((3, None, None))),
            right: build_tree(Some((
                7,
                build_tree(Some((6, None, None))),
                None,
            ))),
        };
        assert!(first_common_ancestor_4_8(&t1, &t2));

        // T3 is not a subtree of T1 (structure doesn't match)
        let t3 = TreeNode {
            value: 5,
            left: build_tree(Some((3, None, None))),
            right: build_tree(Some((
                7,
                None,
                build_tree(Some((6, None, None))),
            ))),
        };
        assert!(!first_common_ancestor_4_8(&t1, &t3));

        // T4 is a single node that exists in T1
        let t4 = TreeNode::new(12);
        assert!(first_common_ancestor_4_8(&t1, &t4));

        // T5 is a single node that does not exist in T1
        let t5 = TreeNode::new(99);
        assert!(!first_common_ancestor_4_8(&t1, &t5));

        // T6 is the whole tree T1 (should be a subtree of itself)
        assert!(first_common_ancestor_4_8(&t1, &t1));

        // T7 is a subtree rooted at 7 with left child 6
        let t7 = TreeNode {
            value: 7,
            left: build_tree(Some((6, None, None))),
            right: None,
        };
        assert!(first_common_ancestor_4_8(&t1, &t7));

        // T8 is a subtree rooted at 15 with only right child 18
        let t8 = TreeNode {
            value: 15,
            left: None,
            right: build_tree(Some((18, None, None))),
        };
        assert!(!first_common_ancestor_4_8(&t1, &t8)); // structure doesn't match (missing left child 12)

        // T9 is a subtree rooted at 15 with both children
        let t9 = TreeNode {
            value: 15,
            left: build_tree(Some((12, None, None))),
            right: build_tree(Some((18, None, None))),
        };
        assert!(first_common_ancestor_4_8(&t1, &t9));
    }

    #[test]
    fn test_insertion_5_1() {
        let n: i32 = -1;
        let m: i32 = 0b0100;
        let result = insertion_5_1(n, m, 4, 7);
        let expected: i32 = 0b1111_1111_1111_1111_1111_1111_0100_1111u32 as i32;
        assert_eq!(expected, result);

        let n = 0b100_0010_0000;
        let m = 0b10011;
        let result = insertion_5_1(n, m, 2, 6);
        let expected = 0b100_0100_1100;
        assert_eq!(expected, result);
    }

    #[test]
    fn test_robot_grid_8_2() {
        use RobotGridCell8_2::*;
        use RobotGridMove8_2::*;

        let grid = vec![
            vec![Accessible, Accessible, Inaccessible],
            vec![Inaccessible, Accessible, Accessible],
            vec![Accessible, Inaccessible, Accessible],
        ];

        let result = robot_grid_8_2(&grid);
        assert_eq!(
            result,
            Some(vec![Right, Down, Right, Down])
        );
    }
}
