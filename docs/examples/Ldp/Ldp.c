#include <stdio.h>
#include <string.h>

typedef struct {
    char* service1;
    char* service2;
} Conflict;

typedef struct {
    char* service;
    int node;
} Deployment;

// Conflicts (shared across models)
Conflict conflicts[] = {
    {"HBI Data", "Web Server"},
    {"Web Server", "Voice Recog."}
};

// Deployments for M1
Deployment model1[] = {
    {"HBI Data", 0},
    {"Web Server", 0},
    {"Voice Recog.", 1}
};

// Deployments for M2
Deployment model2[] = {
    {"HBI Data", 1},
    {"Web Server", 0},
    {"Voice Recog.", 1}
};

// Check for deployment conflicts
void check_conflicts(Deployment* deployments, int dep_len) {
    int found_conflict = 0;
    int conf_len = sizeof(conflicts) / sizeof(Conflict);

    for (int i = 0; i < dep_len; i++) {
        for (int j = i + 1; j < dep_len; j++) {
            if (deployments[i].node == deployments[j].node) {
                for (int k = 0; k < conf_len; k++) {
                    if ((strcmp(deployments[i].service, conflicts[k].service1) == 0 &&
                         strcmp(deployments[j].service, conflicts[k].service2) == 0) ||
                        (strcmp(deployments[i].service, conflicts[k].service2) == 0 &&
                         strcmp(deployments[j].service, conflicts[k].service1) == 0)) {

                        if (!found_conflict) {
                            printf("Conflict detected:\n");
                            found_conflict = 1;
                        }

                        printf(" - %s and %s are both deployed on Node %d\n",
                               deployments[i].service,
                               deployments[j].service,
                               deployments[i].node);
                    }
                }
            }
        }
    }

    if (!found_conflict) {
        printf("Deployment is valid (no conflicts).\n");
    }
}

int main() {
    int useM1 = 0;  // <-- Change to 0 for M2

    if (useM1) {
        printf("Using Model M1:\n");
        check_conflicts(model1, sizeof(model1) / sizeof(Deployment));
    } else {
        printf("Using Model M2:\n");
        check_conflicts(model2, sizeof(model2) / sizeof(Deployment));
    }

    return 0;
}
