```shell
sacct --units=G -a -X -o JobID,UID,Account,State,Partition,TimelimitRaw,Submit,Eligible,Planned,Start,End,Priority,ReqCPUS,ReqMem,ReqNodes --parsable2 -S 2021-01-01 -E 2024-05-01 > until_2024-05-01.csv
```