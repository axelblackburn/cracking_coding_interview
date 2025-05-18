typedef struct Node {
    int value;
    struct Node *next;
} Node;

typedef struct SinglyLinkedList {
    Node *head;
} SinglyLinkedList;

Node *intersection_2_7(SinglyLinkedList *list_a, SinglyLinkedList *list_b);

Node *detect_loop_2_8(SinglyLinkedList *list);
