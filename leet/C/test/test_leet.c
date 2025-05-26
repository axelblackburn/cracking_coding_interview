#include "unity.h"
#include "../leet.h"

void setUp(void) {
}

void tearDown(void) {
}

// Lists that intersect:
// A -> B -> C -> D
// F --------^

// Lists that don't intersect:
// A -> B -> C -> D
// F -> G -> H
void test_intersection_2_7(void) {
    Node D = { .value = 0, .next = NULL };
    Node C = { .value = 1, .next = &D };
    Node B = { .value = 2, .next = &C };
    Node A = { .value = 3, .next = &B };
    SinglyLinkedList list_a = { .head = &A };
    Node F = { .value = 4, .next = &C };
    SinglyLinkedList list_b = { .head = &F };
    TEST_ASSERT_EQUAL(intersection_2_7(&list_a, &list_b), &C);

    Node H = { .value = 6, .next = NULL };
    Node G = { .value = 7, .next = &H };
    F.next = &G;
    TEST_ASSERT_EQUAL(intersection_2_7(&list_a, &list_b), NULL);
}

void test_detect_loop_2_8(void) {
    Node D = { .value = 0, .next = NULL };
    Node C = { .value = 1, .next = &D };
    Node B = { .value = 2, .next = &C };
    Node A = { .value = 3, .next = &B };
    SinglyLinkedList list_a = { .head = &A };
    TEST_ASSERT_EQUAL(detect_loop_2_8(&list_a), NULL);

    D.next = &C;
    TEST_ASSERT_EQUAL(detect_loop_2_8(&list_a), &C);

    D.next = &B;
    TEST_ASSERT_EQUAL(detect_loop_2_8(&list_a), &B);

    D.next = &A;
    TEST_ASSERT_EQUAL(detect_loop_2_8(&list_a), &A);
}

void test_find_next_node_4_6(void) {
    LinkedTreeNode left_right_node = { .value = 3, .parent = NULL, .left = NULL, .right = NULL };
    LinkedTreeNode left_node = { .value = 2, .parent = NULL, .left = NULL, .right = NULL };
    left_node.right = &left_right_node;
    left_right_node.parent = &left_node;

    LinkedTreeNode right_left_node = { .value = 5, .parent = NULL, .left = NULL, .right = NULL };
    LinkedTreeNode right_right_node = { .value = 7, .parent = NULL, .left = NULL, .right = NULL };
    LinkedTreeNode right_node = { .value = 6, .parent = NULL, .left = &right_left_node, .right = &right_right_node };
    right_left_node.parent = &right_node;
    right_right_node.parent = &right_node;

    LinkedTreeNode root = { .value = 4, .parent = NULL, .left = &left_node, .right = &right_node };
    left_node.parent = &root;
    right_node.parent = &root;

    TEST_ASSERT_EQUAL(find_next_node_4_6(&root), &right_left_node);
    TEST_ASSERT_EQUAL(find_next_node_4_6(&right_left_node), &right_node);
    TEST_ASSERT_EQUAL(find_next_node_4_6(&left_right_node), &root);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_intersection_2_7);
    RUN_TEST(test_detect_loop_2_8);
    RUN_TEST(test_find_next_node_4_6);
    return UNITY_END();
}
