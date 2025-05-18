#include "leet.h"

#include <stdbool.h>
#include <stddef.h>


// Problem 2.7: Intersection
// Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting node. Note that the intersection is defined based on reference, not value. That is, if the kth node of the first linked list is the exact same node (by reference) as the jth node of the second linked list, then they are intersecting.

// Lists that intersect:
// A -> B -> C -> D
// E -> F ---^

// Lists that don't intersect:
// A -> B -> C -> D
// E -> F -> G -> H

Node *intersection_2_7(SinglyLinkedList *list_a, SinglyLinkedList *list_b) {
    size_t a_length = 0;
    size_t b_length = 0;

    Node *runner = list_a->head;
    while (runner) {
        runner = runner->next;
        a_length++;
    }

    runner = list_b->head;
    while (runner) {
        runner = runner->next;
        b_length++;
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
