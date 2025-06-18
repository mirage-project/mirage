use egg::{rewrite as rw, *};
use std::{time::Duration};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
pub type EGraph = egg::EGraph<Expr, ()>;
pub type Rewrite = egg::Rewrite<Expr, ()>;

define_language! {
    pub enum Expr {
        "ew_add" = EWAdd([Id; 2]),
        "ew_mul" = EWMul([Id; 2]),
        "bc_div" = BDDiv([Id; 2]),
        "bc_pow" = BCPow([Id; 2]),

        "concat" = Concat([Id; 3]),
        "ew_exp" = EWExp(Id),
        "square" = Square(Id),
        "sqrt" = Sqrt(Id),
        "matmul" = MatMul([Id; 2]),
        "sum" = Sum([Id; 2]),
        "mean" = Mean(Id),

        "rms" = Rms(Id),
        "rms_norm" = RmsNorm(Id),
        "silu" = Silu(Id),
        "gelu" = Gelu(Id),
        "relu" = Relu(Id),
        "clamp" = Clamp(Id),
        "partition" = Partition([Id; 4]),
        "combine" = Combine([Id;3]),
        "replicate" = Replicate([Id; 3]),
        "reduce" = Reduce([Id; 2]),

        Symbol(Symbol),
    }
}

fn is_equal(vars: &[&str]) -> impl Fn(&mut EGraph, Id, &Subst) -> bool + 'static {
    let parsed_vars: Vec<Var> = vars.iter().map(|v| v.parse().unwrap()).collect();

    move |egraph, _, subst| {
        let mut strings = Vec::new();
        for var in &parsed_vars {
            let id = subst[*var];
            let node = &egraph[id];

            let symbol_opt = node.nodes.iter().find_map(|n| {
                if let Expr::Symbol(s) = n {
                    Some(s.clone())
                } else {
                    None
                }
            });

            match symbol_opt {
                Some(s) => strings.push(s),
                None => return false,
            }
        }

        for i in 0..strings.len() {
            for j in (i + 1)..strings.len() {
                if strings[i] != strings[j] {
                    return false;
                }
            }
        }

        true
    }
}

fn is_reddim(vars: &[&str]) -> impl Fn(&mut EGraph, Id, &Subst) -> bool + 'static {
    let parsed_vars: Vec<Var> = vars.iter().map(|v| v.parse().unwrap()).collect();

    move |egraph, _, subst| {
        let mut strings = Vec::new();
        for var in &parsed_vars {
            let id = subst[*var];
            let node = &egraph[id];

            let symbol_opt = node.nodes.iter().find_map(|n| {
                if let Expr::Symbol(s) = n {
                    Some(s.clone())
                } else {
                    None
                }
            });

            match symbol_opt {
                Some(s) => strings.push(s),
                None => return false,
            }
        }

        for i in 0..strings.len() {
            for j in (i + 1)..strings.len() {
                if strings[i] != strings[j] || strings[i] != "reddim0".into() {
                    return false;
                }
            }
        }

        true
    }
}


fn is_unique(vars: &[&str]) -> impl Fn(&mut EGraph, Id, &Subst) -> bool + 'static {
    assert!(vars.len() == 3, "is_unique expects exactly 3 variables");
    let parsed_vars: Vec<Var> = vars.iter().map(|v| v.parse().unwrap()).collect();

    move |egraph, _, subst| {
        let mut strings = Vec::new();
        for var in &parsed_vars {
            let id = subst[*var];
            let node = &egraph[id];

            let symbol_opt = node.nodes.iter().find_map(|n| {
                if let Expr::Symbol(s) = n {
                    Some(s.clone())
                } else {
                    None
                }
            });

            match symbol_opt {
                Some(s) => strings.push(s),
                None => return false,
            }
        }

        for i in 0..strings.len() {
            for j in (i + 1)..strings.len() {
                if strings[i] == strings[j] {
                    return false;
                }
            }
        }

        true
    }
}

fn is_biunique(vars: &[&str]) -> impl Fn(&mut EGraph, Id, &Subst) -> bool + 'static {
    assert!(vars.len() == 4, "is_biunique expects exactly 4 variables");
    let parsed_vars: Vec<Var> = vars.iter().map(|v| v.parse().unwrap()).collect();

    move |egraph, _, subst| {
        let mut symbols = Vec::new();
        for var in &parsed_vars {
            let id = subst[*var];
            let node = &egraph[id];

            let symbol_opt = node.nodes.iter().find_map(|n| {
                if let Expr::Symbol(s) = n {
                    Some(s.clone())
                } else {
                    None
                }
            });

            match symbol_opt {
                Some(s) => symbols.push(s),
                None => return false,
            }
        }

        if symbols[0] == symbols[2] || symbols[0] == symbols[3] {
            return false;
        }

        if symbols[1] == symbols[2] || symbols[1] == symbols[3] {
            return false;
        }

        true
    }
}

#[rustfmt::skip]
pub fn rules() -> Vec<Rewrite> {
    let mut rules = vec![
        rw!("ew_add_comm"; 
            "(ew_add ?t0 ?t1)" 
            <=> "(ew_add ?t1 ?t0)"),

        rw!("ew_mul_comm"; 
            "(ew_mul ?t0 ?t1)" 
            <=> "(ew_mul ?t1 ?t0)"),

        rw!("ew_add_assoc"; 
            "(ew_add ?t0 (ew_add ?t1 ?t2))" 
            <=> "(ew_add (ew_add ?t0 ?t1) ?t2)"),

        rw!("ew_mul_assoc"; 
            "(ew_mul ?t0 (ew_mul ?t1 ?t2))" 
            <=> "(ew_mul (ew_mul ?t0 ?t1) ?t2)"),

        rw!("matmul_assoc"; 
            "(matmul ?t0 (matmul ?t1 ?t2))" 
            <=> "(matmul (matmul ?t0 ?t1) ?t2)"),

        rw!("ew_mul_distrib_add"; 
            "(ew_mul (ew_add ?t0 ?t1) ?t2)" 
            <=> "(ew_add (ew_mul ?t0 ?t2) (ew_mul ?t1 ?t2))"),

        rw!("bc_div_distrib_add"; 
            "(bc_div (ew_add ?t0 ?t1) ?t2)" 
            <=> "(ew_add (bc_div ?t0 ?t2) (bc_div ?t1 ?t2))"),

        rw!("matmul_distrib_add_left"; 
            "(matmul (ew_add ?t0 ?t1) ?t2)" 
            <=> "(ew_add (matmul ?t0 ?t2) (matmul ?t1 ?t2))"),

        rw!("matmul_distrib_add_right"; 
            "(matmul ?t0 (ew_add ?t1 ?t2))" 
            <=> "(ew_add (matmul ?t0 ?t1) (matmul ?t0 ?t2))"),

        rw!("matmul_bc_div"; 
            "(matmul (bc_div ?t0 ?t1) ?t2)" 
            <=> "(bc_div (matmul ?t0 ?t2) ?t1)"),

        rw!("bc_pow_distrib_mul"; 
            "(bc_pow (ew_mul ?t0 ?t1) ?t2)" 
            <=> "(ew_mul (bc_pow ?t0 ?t2) (bc_pow ?t1 ?t2))"),

        rw!("bc_pow_distrib_add"; 
            "(bc_pow ?t0 (ew_add ?t1 ?t2))" 
            <=> "(ew_mul (bc_pow ?t0 ?t1) (bc_pow ?t0 ?t2))"),

        rw!("rms_norm_to_rms";
            "(rms_norm ?t0)"
            <=> "(bc_div ?t0 (rms ?t0))"),
    ].concat();

    let mut rules1 = vec![
        rw!("partition_reduce"; 
            "(partition (reduce ?t0 ?d0) ?d1 ?d2 ?i0)" 
            <=> "(reduce (partition ?t0 ?d1 ?d2 ?i0) ?d0)" 
            if is_unique(&["?d0", "?d1", "?d2"]) ),

        rw!("combine_reduce"; 
            "(combine (reduce ?t0 ?d0) ?d1 ?d2)" 
            <=> "(reduce (combine ?t0 ?d1 ?d2) ?d0)" 
            if is_unique(&["?d0", "?d1", "?d2"]) ),

        rw!("partition_replicate"; 
            "(partition (replicate ?t0 ?d0 ?i0) ?d1 ?d2 ?i1)" 
            <=> "(replicate (partition ?t0 ?d1 ?d2 ?i1) ?d0 ?i0)" 
            if is_unique(&["?d0", "?d1", "?d2"]) ),

        rw!("combine_replicate"; 
            "(combine (replicate ?t0 ?d0 ?i0) ?d1 ?d2)" 
            <=> "(replicate (combine ?t0 ?d1 ?d2) ?d0 ?i0)" 
            if is_unique(&["?d0", "?d1", "?d2"]) ),

        rw!("partition_sum"; 
            "(partition (sum ?t0 ?d0) ?d1 ?d2 ?i0)" 
            <=> "(sum (partition ?t0 ?d1 ?d2 ?i0) ?d0)" 
            if is_unique(&["?d0", "?d1", "?d2"]) ),

        rw!("combine_sum"; 
            "(combine (sum ?t0 ?d0) ?d1 ?d2)" 
            <=> "(sum (combine ?t0 ?d1 ?d2) ?d0)" 
            if is_unique(&["?d0", "?d1", "?d2"]) ),
    ].concat();

    let mut rules2 = vec![
        rw!("partition-partition-commute"; 
            "(partition (partition ?t0 ?d0 ?d1 ?i0) ?d2 ?d3 ?i1)" 
            <=> "(partition (partition ?t0 ?d2 ?d3 ?i1) ?d0 ?d1 ?i0)" 
            if is_biunique(&["?d0", "?d1", "?d2", "?d3"]) ),

        rw!("combine-combine-commute"; 
            "(combine (combine ?t0 ?d0 ?d1) ?d2 ?d3)" 
            <=> "(combine (combine ?t0 ?d2 ?d3) ?d0 ?d1)" 
            if is_biunique(&["?d0", "?d1", "?d2", "?d3"]) ),
    ].concat();

    let mut rules3 = vec![
        rw!("reduce_sum_idempotent"; 
            "(reduce (sum ?t0 ?d0) ?d1)" 
            => "(reduce ?t0 ?d0)" 
            if is_equal(&["?d0", "?d1"]) ),

        rw!("partition_combine_cancel"; 
            "(combine (partition ?t0 ?d0 ?d1 ?i0) ?d2 ?d3)" 
            => "?t0" if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"]) ),

        rw!("combine_partition_cancel"; 
            "(partition (combine ?t0 ?d0 ?d1) ?d2 ?d3 ?i0)" 
            => "?t0" if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"]) ),

        rw!("matmul-partition-combine"; 
            "(combine (matmul (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d2 ?d3 ?i1)) ?d4 ?d5)" 
            => "(matmul ?t0 ?t1)" 
            if is_equal(&["?d0", "?d2", "?d4"]) 
            if is_equal(&["?d1", "?d3", "?d5"]) 
            if is_equal(&["?i0", "?i1"]) ),

        rw!("col-partitioned-matmul"; 
            "(combine (matmul (replicate ?t0 ?d0 ?i0) (partition ?t1 ?d1 ?d2 ?i1)) ?d3 ?d4)" 
            => "(matmul ?t0 ?t1)" 
            if is_equal(&["?d0", "?d2", "?d4"]) 
            if is_equal(&["?d1", "?d3"]) 
            if is_equal(&["?i0", "?i1"]) ),

        rw!("col-replicated-rms";
            "(combine (rms (replicate ?t0 ?d0 ?i0)) ?d1 ?d2)" 
            => "(rms ?t0)" 
            if is_equal(&["?d0", "?d2"]) ),

        rw!("col-partitioned-rms";
            "(rms (reduce (partition ?t0 ?d0 ?d1 ?i0) ?d2))" 
            => "(rms ?t0)" 
            if is_equal(&["?d1", "?d2"]) ),

        rw!("partial-sum-matmul"; 
            "(reduce (matmul (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d2 ?d3 ?i1)) ?d4)" 
            => "(matmul ?t0 ?t1)" 
            if is_equal(&["?d1", "?d3", "?d4"]) 
            if is_equal(&["?i0", "?i1"]) ),
    ];

    let mut rules4 = vec![
        rw!("partitioned-matmul"; 
            "(partition (matmul ?t0 ?t1) ?d1 ?d0 ?i0)" 
            <=> "(matmul (partition ?t0 ?d1 ?d0 ?i0) (partition ?t1 ?d1 ?d0 ?i0))"),

        rw!("partitioned-matmul2"; 
            "(partition (matmul ?t0 ?t1) ?d0 ?d1 ?i0)" 
            <=> "(matmul (partition ?t0 ?d0 ?d1 ?i0) (replicate ?t1 ?d1 ?i0))"),
    ].concat();

   let mut rules5 = vec![
        rw!("ew-add-combine"; 
            "(combine (ew_add (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d2 ?d3 ?i1)) ?d4 ?d5)" 
            => "(ew_add ?t0 ?t1)" 
            if is_equal(&["?d0", "?d2", "?d4"]) 
            if is_equal(&["?d1", "?d3", "?d5"]) 
            if is_equal(&["?i0", "?i1"]) ),

        rw!("ew-mul-combine";
            "(combine (ew_mul (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d2 ?d3 ?i1)) ?d4 ?d5)" 
            => "(ew_mul ?t0 ?t1)" 
            if is_equal(&["?d0", "?d2", "?d4"]) 
            if is_equal(&["?d1", "?d3", "?d5"]) 
            if is_equal(&["?i0", "?i1"])
        ),

        rw!("bc-pow-combine";
            "(combine (bc_pow (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d2 ?d3 ?i1)) ?d4 ?d5)" 
            => "(bc_pow ?t0 ?t1)" 
            if is_equal(&["?d0", "?d2", "?d4"]) 
            if is_equal(&["?d1", "?d3", "?d5"]) 
            if is_equal(&["?i0", "?i1"])
        ),

        rw!("reduce-exp";
            "(reduce (sum (ew_exp (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (ew_exp ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),

        rw!("reduce-rms-norm";
            "(reduce (sum (rms_norm (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (rms_norm ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),

        rw!("reduce-rms";
            "(reduce (sum (rms (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (rms ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),

        rw!("reduce-square";
            "(reduce (sum (square (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (square ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),

        rw!("reduce-sqrt";
            "(reduce (sum (sqrt (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (sqrt ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),

        rw!("reduce-silu";
            "(reduce (sum (silu (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (silu ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),

        rw!("reduce-gelu";
            "(reduce (sum (gelu (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (gelu ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),

        rw!("reduce-relu";
            "(reduce (sum (relu (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (relu ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),

        rw!("reduce-clamp";
            "(reduce (sum (clamp (partition ?t0 ?d0 ?d1 ?i0)) ?d2) ?d3)" 
            => "(reduce (clamp ?t0) ?d0)" 
            if is_equal(&["?d0", "?d2"]) 
            if is_equal(&["?d1", "?d3"])
        ),
    ]; 

    let mut rules6 = vec![
        rw!("ew-add-partition"; 
            "(partition (ew_add ?t0 ?t1) ?d0 ?d1 ?i0)" 
            <=> "(ew_add (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0))"),

        rw!("ew-mul-partition"; 
            "(partition (ew_mul ?t0 ?t1) ?d0 ?d1 ?i0)" 
            <=> "(ew_mul (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0))"),

        rw!("bc-div-partition"; 
            "(partition (bc_div ?t0 ?t1) ?d0 ?d1 ?i0)" 
            <=> "(bc_div (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0))"),

        rw!("bc-pow-partition"; 
            "(partition (bc_pow ?t0 ?t1) ?d0 ?d1 ?i0)" 
            <=> "(bc_pow (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0))"),

        rw!("ew-add-combine2"; 
            "(combine (ew_add ?t0 ?t1) ?d0 ?d1)" 
            <=> "(ew_add (combine ?t0 ?d0 ?d1) (combine ?t1 ?d0 ?d1))"),

        rw!("ew-mul-combine2"; 
        "(combine (ew_mul ?t0 ?t1) ?d0 ?d1)" 
        <=> "(ew_mul (combine ?t0 ?d0 ?d1) (combine ?t1 ?d0 ?d1))"),

        rw!("bc-div-combine2"; 
            "(combine (bc_div ?t0 ?t1) ?d0 ?d1)" 
            <=> "(bc_div (combine ?t0 ?d0 ?d1) (combine ?t1 ?d0 ?d1))"),

        rw!("bc-pow-combine2"; 
            "(combine (bc_pow ?t0 ?t1) ?d0 ?d1)" 
            <=> "(bc_pow (combine ?t0 ?d0 ?d1) (combine ?t1 ?d0 ?d1))"),

        rw!("ew-add-replicate"; 
            "(replicate (ew_add ?t0 ?t1) ?d ?i0)" 
            <=> "(ew_add (replicate ?t0 ?d ?i0) (replicate ?t1 ?d ?i0))"),

        rw!("ew-mul-replicate"; 
            "(replicate (ew_mul ?t0 ?t1) ?d ?i0)" 
            <=> "(ew_mul (replicate ?t0 ?d ?i0) (replicate ?t1 ?d ?i0))"),

        rw!("bc-div-replicate"; 
            "(replicate (bc_div ?t0 ?t1) ?d ?i0)" 
            <=> "(bc_div (replicate ?t0 ?d ?i0) (replicate ?t1 ?d ?i0))"),

        rw!("bc-pow-replicate"; 
            "(replicate (bc_pow ?t0 ?t1) ?d ?i0)" 
            <=> "(bc_pow (replicate ?t0 ?d ?i0) (replicate ?t1 ?d ?i0))"),

        rw!("partition-exp"; 
            "(partition (ew_exp ?t0) ?d0 ?d1 ?i0)" 
            <=> "(ew_exp (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-rms-norm"; 
            "(partition (rms_norm ?t0) ?d0 ?d1 ?i0)" 
            <=> "(rms_norm (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-rms";
            "(partition (rms ?t0) ?d0 ?d1 ?i0)" 
            <=> "(rms (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-square"; 
            "(partition (square ?t0) ?d0 ?d1 ?i0)" 
            <=> "(square (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-sqrt"; 
            "(partition (sqrt ?t0) ?d0 ?d1 ?i0)" 
            <=> "(sqrt (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-silu"; 
            "(partition (silu ?t0) ?d0 ?d1 ?i0)" 
            <=> "(silu (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-gelu"; 
            "(partition (gelu ?t0) ?d0 ?d1 ?i0)" 
            <=> "(gelu (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-relu"; 
            "(partition (relu ?t0) ?d0 ?d1 ?i0)" 
            <=> "(relu (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-clamp"; 
            "(partition (clamp ?t0) ?d0 ?d1 ?i0)" 
            <=> "(clamp (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("combine-exp"; 
            "(combine (ew_exp ?t0) ?d0 ?d1)" 
            <=> "(ew_exp (combine ?t0 ?d0 ?d1))"),

        rw!("combine-square"; 
            "(combine (square ?t0) ?d0 ?d1)" 
            <=> "(square (combine ?t0 ?d0 ?d1))"),

        rw!("combine-sqrt"; 
            "(combine (sqrt ?t0) ?d0 ?d1)" 
            <=> "(sqrt (combine ?t0 ?d0 ?d1))"),

        rw!("combine-silu"; 
            "(combine (silu ?t0) ?d0 ?d1)"
            <=> "(silu (combine ?t0 ?d0 ?d1))"),

        rw!("combine-gelu"; 
            "(combine (gelu ?t0) ?d0 ?d1)" 
            <=> "(gelu (combine ?t0 ?d0 ?d1))"),

        rw!("combine-relu"; 
            "(combine (relu ?t0) ?d0 ?d1)" 
            <=> "(relu (combine ?t0 ?d0 ?d1))"),

        rw!("combine-clamp"; 
            "(combine (clamp ?t0) ?d0 ?d1)" 
            <=> "(clamp (combine ?t0 ?d0 ?d1))"),

        rw!("replicate-exp"; 
            "(replicate (ew_exp ?t0) ?d ?i0)" 
            <=> "(ew_exp (replicate ?t0 ?d ?i0))"),

        rw!("replicate-rms-norm";
            "(replicate (rms_norm ?t0) ?d ?i0)" 
            <=> "(rms_norm (replicate ?t0 ?d ?i0))"),

        rw!("replicate-rms";
            "(replicate (rms ?t0) ?d ?i0)" 
            <=> "(rms (replicate ?t0 ?d ?i0))"),

        rw!("replicate-square"; 
            "(replicate (square ?t0) ?d ?i0)" 
            <=> "(square (replicate ?t0 ?d ?i0))"),

        rw!("replicate-sqrt"; 
            "(replicate (sqrt ?t0) ?d ?i0)" 
            <=> "(sqrt (replicate ?t0 ?d ?i0))"),

        rw!("replicate-silu"; 
            "(replicate (silu ?t0) ?d ?i0)" 
            <=> "(silu (replicate ?t0 ?d ?i0))"),

        rw!("replicate-gelu"; 
            "(replicate (gelu ?t0) ?d ?i0)" 
            <=> "(gelu (replicate ?t0 ?d ?i0))"),

        rw!("replicate-relu"; 
            "(replicate (relu ?t0) ?d ?i0)" 
            <=> "(relu (replicate ?t0 ?d ?i0))"),

        rw!("replicate-clamp"; 
            "(replicate (clamp ?t0) ?d ?i0)" 
            <=> "(clamp (replicate ?t0 ?d ?i0))"),

    ].concat();

    rules.append(&mut rules1);
    rules.append(&mut rules2);
    rules.append(&mut rules3);
    rules.append(&mut rules4);
    rules.append(&mut rules5);
    rules.append(&mut rules6);

    rules
}

#[no_mangle]
pub extern "C" fn check_equiv(expr1: *const c_char, expr2: *const c_char) -> bool {
    let expr1_str: &str = unsafe {
            CStr::from_ptr(expr1)
        }.to_str().unwrap_or("");

    let expr2_str: &str = unsafe {
            CStr::from_ptr(expr2)
        }.to_str().unwrap_or("");

    let runner = Runner::default()
    .with_iter_limit(100_000)
    .with_node_limit(100_000)
    .with_time_limit(Duration::from_secs(1000))
    .with_expr(&expr2_str.parse().unwrap())
    .run(&rules());
    
    let egraph = &runner.egraph;

    let id1 = egraph.lookup_expr(&expr1_str.parse().unwrap());

    id1.is_some()

}
