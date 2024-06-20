```shell
sacct -a -X -o JobID,UID,Account,State,Partition,TimelimitRaw,Submit,Eligible,ElapsedRaw,Planned,Start,End,Priority,ReqCPUS,ReqMem,ReqNodes,ReqTRES,QOS --parsable2 -S 2021-01-01 -E 2024-05-01 > until_2024-05-01.csv
```

```shell
cd /anvil/projects/x-cda090008/trout
```
