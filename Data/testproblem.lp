\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  5 C0 + 3 C1
Subject To
 R0: C0 + 5 C1 - 3 C2 <= -0
 R1: 4 C0 + 3 C1 - C2 <= 5
 qc0: 3 C1 - C2 + [ 4 C0 ^2 ] <= -0
Bounds
 C0 <= 2
 C1 <= 30
 C2 <= 30
End
