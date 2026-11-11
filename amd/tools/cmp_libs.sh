# cmp_libs.sh NEW REF: stripped lib.s of every (kind, T) in NEW vs REF
st() { [ -f $1 ] && grep -v '^\s*\.\(loc\|file\|ident\)\|^\s*;\|hip_cuid' $1 | sed "s/;.*//" | md5sum | cut -c1-12 || echo MISSING; }
n=0; m=0
for k in kdad mla moe moeagg moeaggkdad moeaggmla; do for t in 1 2 4 8 16 32 64; do
  a=$(st /mpk-amd/$1/${k}_t$t/lib.s); b=$(st $(readlink -f /mpk-amd/$2/${k}_t$t)/lib.s); n=$((n+1))
  if [ "$a" = "$b" ]; then m=$((m+1)); else echo "DIFF ${k}_t$t new $a ref $b"; fi; done; done
echo "SAME $m of $n"
