#include "leet.h"

#include <stdbool.h>
#include <stddef.h>


// Problem 2.7: Intersection
// Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting node. Note that the intersection is defined based on reference, not value. That is, if the kth node of the first linked list is the exact same node (by reference) as the jth node of the second linked list, then they are intersecting.

// Lists that intersect:
// A -> B -> C -> D
// F --------^

// Lists that don't intersect:
// A -> B -> C -> D
// F -> G -> H

Node *intersection_2_7(SinglyLinkedList *list_a, SinglyLinkedList *list_b) {
    size_t a_length = 0;
    size_t b_length = 0;

    Node *runner = list_a->head;
    Node *a_tail = NULL;
    Node *b_tail = NULL;
    while (runner) {
        a_tail = runner;
        runner = runner->next;
        a_length++;
    }

    runner = list_b->head;
    while (runner) {
        b_tail = runner;
        runner = runner->next;
        b_length++;
    }

    if (a_tail != b_tail) {
        return NULL;
    }

    Node *longest_runner;
    Node *shortest_runner;
    size_t length_diff;
    if (a_length > b_length) {
        longest_runner = list_a->head;
        shortest_runner = list_b->head;
        length_diff = a_length - b_length;
    } else {
        longest_runner = list_b->head;
        shortest_runner = list_a->head;
        length_diff = b_length - a_length;
    }

    for (int i = 0; i < length_diff; i++) {
        longest_runner = longest_runner->next;
    }

    while (longest_runner) {
        if (longest_runner == shortest_runner) {
            return longest_runner;
        }
        longest_runner = longest_runner->next;
        shortest_runner = shortest_runner->next;
    }

    return NULL;
}

// Problem 2.8: Loop Detection
// Given a circular linked list, implement an algorithm that returns the node at the beginning of the loop.

// 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9

// If 9 -> 1:
// SF
//      S    F
//           S         F
//                S              F
//                     S                   F
//      F                   S
//                F              S
//                          F         S
//                                    F     S
// SF

// If 9 -> 2:
// SF
//      S    F
//           S         F
//                S              F
//                     S                   F
//           F              S
//                     F         S
//                               F    S
//                                         SF

// If 9 -> 3
// SF
//      S    F
//           S         F
//                S              F
//                     S                   F
//                F         S
//                          F    S
//                                    SF

// If 9 -> 4
// SF
//      S    F
//           S         F
//                S              F
//                     S                   F
//                     F    S
//                               SF


// Collision occurs N steps before the start of the loop, where N is also the distance between the list and the start of the loop
Node *detect_loop_2_8(SinglyLinkedList *list) {
    Node *slow_runner = list->head;
    Node *fast_runner = slow_runner;

    if (slow_runner->next != NULL) {
        slow_runner = slow_runner->next;
        fast_runner = fast_runner->next;

        if (fast_runner->next != NULL) {
            fast_runner = fast_runner->next;
        } else {
            return NULL;
        }
    } else {
        return NULL;
    }

    while (fast_runner != NULL && slow_runner != fast_runner) {
        if (fast_runner->next != NULL) {
            slow_runner = slow_runner->next;
            fast_runner = fast_runner->next->next;
        } else {
            return NULL;
        }
    }

    if (fast_runner == NULL) {
        return NULL;
    }

    slow_runner = list->head;
    while (slow_runner != fast_runner) {
        slow_runner = slow_runner->next;
        fast_runner = fast_runner->next;
    }

    return slow_runner;
}

// Problem 4.6: Successor
// Write an algorithm to find the "next" node (i.e., in-order successor) of a given node in a binary search tree. 
// You may assume that each node has a link to its parent.

const LinkedTreeNode * get_upmost_left_child(const LinkedTreeNode *node) {
    if (!node) {
        return NULL;
    }

    if (node->left) {
        return get_upmost_left_child(node->left);
    }

    return node;
}

const LinkedTreeNode * find_next_node_4_6(const LinkedTreeNode *node) {
    if (!node) {
        return NULL;
    }

    if (node->right) {
        return get_upmost_left_child(node->right);
    }

    // If the node has no right child, traverse up the tree until we find a node that is a left child of its parent.
    const LinkedTreeNode* current = node;
    const LinkedTreeNode* parent = node->parent;
    while (parent && parent->right == current) {
        current = parent;
        parent = parent->parent;
    }
    return parent;
}
