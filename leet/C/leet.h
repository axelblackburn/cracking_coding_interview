typedef struct Node {
    int value;
    struct Node *next;
} Node;

typedef struct SinglyLinkedList {
    Node *head;
} SinglyLinkedList;

Node *intersection_2_7(SinglyLinkedList *list_a, SinglyLinkedList *list_b);

Node *detect_loop_2_8(SinglyLinkedList *list);

typedef struct LinkedTreeNode {
    int value;
    struct LinkedTreeNode *parent;
    struct LinkedTreeNode *left;
    struct LinkedTreeNode *right;
} LinkedTreeNode;

const LinkedTreeNode * find_next_node_4_6(const LinkedTreeNode *node);
