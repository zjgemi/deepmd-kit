#!/bin/bash

output=$(lbg job submit -i params.json -p ./ -oji)
if [ -f "jid" ]; then
    rm "jid"
fi
job_id=$(echo "$output" | tail -n 1)
echo "$job_id" > jid