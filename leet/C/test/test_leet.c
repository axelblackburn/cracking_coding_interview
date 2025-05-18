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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_intersection_2_7);
    return UNITY_END();
}
