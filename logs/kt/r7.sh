#!/bin/sh
cd ~/strix-rt; sh logs/kt/ab.sh "KT_COND=0" "KT_COND=1"; MAXWAIT=60 sh logs/kt/kt_e2e.sh
