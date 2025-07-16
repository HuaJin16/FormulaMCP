#include <stdio.h>
#include <stdbool.h>

bool is_unique(int digits[], int len) {
    for (int i = 0; i < len; i++)
        for (int j = i + 1; j < len; j++)
            if (digits[i] == digits[j])
                return false;
    return true;
}

int main() {
    int S, E, N, D, M, O, R, Y;
    int digits[8];

    for (S = 1; S <= 9; S++)
    for (E = 0; E <= 9; E++)
    for (N = 0; N <= 9; N++)
    for (D = 0; D <= 9; D++)
    for (M = 1; M <= 9; M++)
    for (O = 0; O <= 9; O++)
    for (R = 0; R <= 9; R++)
    for (Y = 0; Y <= 9; Y++) {
        digits[0] = S; digits[1] = E; digits[2] = N; digits[3] = D;
        digits[4] = M; digits[5] = O; digits[6] = R; digits[7] = Y;

        if (!is_unique(digits, 8)) continue;

        int SEND = 1000*S + 100*E + 10*N + D;
        int MORE = 1000*M + 100*O + 10*R + E;
        int MONEY = 10000*M + 1000*O + 100*N + 10*E + Y;

        if (SEND + MORE == MONEY) {
            printf("  SEND = %d\n", SEND);
            printf("  MORE = %d\n", MORE);
            printf(" MONEY = %d\n", MONEY);
            printf("Mapping: S=%d E=%d N=%d D=%d M=%d O=%d R=%d Y=%d\n",
                   S,E,N,D,M,O,R,Y);
        }
    }

    return 0;
}
