## in-vitro experiments taken from DNA-Aeon:

The sleeping beauty novel has been encoded using three general approaches (as described in the DNA-Aeon paper).
We used "Dorn3" and "Dorn5" for our experiments.
The raw data was randomly sub-sampled to produce multiple outputs of differing coverage and processed using
RepairNatrix ( [https://doi.org/10.1093/bioadv/vbad117](https://doi.org/10.1093/bioadv/vbad117)).

Naming convention:
Dorn[X]_DNA_Aeon\_(247)\_[Y]\_(redQ)[Z].ini/.zip:

- [X] = 3 or 5 - the experiment used from DNA-Aeon (spacing between the DNA-Aeon checkpoints)
- 247 = number of packets initially generated
- [Y] = 1-5 - the coverage of the data (reduced to yy.yy% of the original data)
- redQ = reduced quality reads filtered
- [Z] = optional description or parameters for RepairNatrix

It can be seen that for a coverage of 0.70% and optimal preprocessing, the decoding fails.
However, using our approach it is possible to recover the original data:

- Find all missing rows using "Analyze" (MissingRowRepair)
- Repair row content of row 0 (Header Chunk) with an all zero row by pressing "Automatic Repair"
- As most of the missing data of this row will be 00, the result will be mostly correct
- Coomit the added rows to the initial GEPP using the appropriate button
- After this, we can either manually tag the rows still corrupt or use "Tag affected chunks as invalid" with the packet
  id of the newly added row (it will be the last index!)
- We can then select a row to repair and "Open (the) repair window" to manually repair any of the corrupr rows.

Alternatively, we can use the "LangaugeToolTextRepair" to try an automatic repair of the rows still corrupt.

More information about DNA-Aeon can be found at:

- [https://github.com/MW55/DNA-Aeon](https://github.com/MW55/DNA-Aeon)
- [https://doi.org/10.1038/s41467-023-36297-3](https://doi.org/10.1038/s41467-023-36297-3)