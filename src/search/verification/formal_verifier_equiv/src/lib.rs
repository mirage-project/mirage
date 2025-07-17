use egg::{rewrite as rw, *};
use std::{time::Duration};
use std::ffi::CStr;
use std::os::raw::{c_char};
use regex::Regex;

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

        "rms" = Rms([Id; 2]),
        "rms_norm" = RmsNorm([Id; 2]),

        "silu" = Silu(Id),
        "gelu" = Gelu(Id),
        "relu" = Relu(Id),
        "clamp" = Clamp(Id),
        "partition" = Partition([Id; 4]),
        "combine" = Combine([Id;3]),
        "replicate" = Replicate([Id; 3]),
        "reduce" = Reduce([Id; 2]),
        "partial_sum" = PartialSum([Id; 3]),
        "dim_mul" = DimMul([Id; 2]),
         Symbol(Symbol),
    }
}


fn is_unique(vars: &[&str]) -> impl Fn(&mut EGraph, Id, &Subst) -> bool + 'static {
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

fn is_datadim(vars: &[&str], data_dims: Vec<String>) -> impl Fn(&mut EGraph, Id, &Subst) -> bool + 'static {
    let parsed_vars: Vec<Var> = vars.iter().map(|v| v.parse().unwrap()).collect();

    move |egraph, _, subst| {
        let mut strings = Vec::new();

        for var in &parsed_vars {
            let id = subst[*var];
            let node = &egraph[id];
            let symbol = node.nodes.iter().find_map(|n| {
                if let Expr::Symbol(s) = n {
                    Some(s.clone())
                } else {
                    None
                }
            });

            match symbol {
                Some(s) => strings.push(s),
                None => return false,
            }
        }

        if data_dims.iter().any(|s| s == strings[0].as_str()) {
            return true;
        }

        false
    }
}


#[rustfmt::skip]
pub fn rules(mut nums: Vec<u32>) -> Vec<Rewrite> {
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
            "(rms_norm ?t0 ?d0)"
            <=> "(bc_div ?t0 (rms ?t0 ?d0))"),

        rw!("rms_definition";
            "(rms ?t0 ?d0)"
            <=> "(sqrt (sum (square ?t0) ?d0))"),
    ].concat();

    let mut rules1 = vec![

        rw!("partition_replicate"; 
            "(partition (replicate ?t0 ?d0 ?i0) ?d1 ?d2 ?i1)" 
            <=> "(replicate (partition ?t0 ?d1 ?d2 ?i1) ?d0 ?i0)" 
            if is_unique(&["?d0", "?d2"]) ),

        rw!("combine_replicate"; 
            "(combine (replicate ?t0 ?d0 ?i0) ?d1 ?d2)" 
            <=> "(replicate (combine ?t0 ?d1 ?d2) ?d0 ?i0)" 
            if is_unique(&["?d0", "?d2"]) ),

    ].concat();

    let mut rules2 = vec![
        rw!("partition_reduce"; 
            "(reduce (partition ?t0 ?d1 ?d2 ?i0) ?d0)" 
            => "(partition (reduce ?t0 ?d0) ?d1 ?d2 ?i0)" 
            if is_unique(&["?d0", "?d2"])
            if is_datadim(&["?d1"], vec!["data_dim2".to_string(), "data_dim1".to_string()]) ),

        rw!("combine_reduce"; 
            "(combine (reduce ?t0 ?d0) ?d1 ?d2)" 
            => "(reduce (combine ?t0 ?d1 ?d2) ?d0)" 
            if is_unique(&["?d0", "?d2"])
            if is_datadim(&["?d1"], vec!["data_dim2".to_string(), "data_dim1".to_string()]) ),
        
        rw!("partition-partition-commute"; 
            "(partition (partition ?t0 ?d0 ?d1 ?i0) ?d2 ?d3 ?i1)" 
            => "(partition (partition ?t0 ?d2 ?d3 ?i1) ?d0 ?d1 ?i0)" 
            if is_unique(&["?d0", "?d2"])
            if is_unique(&["?d1", "?d3"]) ),

        rw!("combine-combine-commute"; 
            "(combine (combine ?t0 ?d0 ?d1) ?d2 ?d3)" 
            => "(combine (combine ?t0 ?d2 ?d3) ?d0 ?d1)" 
            if is_unique(&["?d0", "?d2"])
            if is_unique(&["?d1", "?d3"]) ),

        rw!("partition_combine_cancel"; 
            "(combine (partition ?t0 ?d0 ?d1 ?i0) ?d0 ?d1)" 
            => "?t0" ),

        rw!("reduce_partial_sum_cancel";
            "(reduce (partial_sum ?t0 ?d0 ?i0) ?d0)" 
            => "?t0" ),

        rw!("partition_sum"; 
            "(sum (partition ?t0 ?d1 ?d2 ?i0) ?d0)"
            => "(partition (sum ?t0 ?d0) ?d1 ?d2 ?i0)"  
            if is_unique(&["?d0", "?d1"]) ),

        rw!("replicate_sum"; 
            "(sum (replicate ?t0 ?d1 ?i0) ?d0)"
            => "(replicate (sum ?t0 ?d0) ?d1 ?i0)" ),

        rw!("combine_sum"; 
            "(combine (sum ?t0 ?d0) ?d1 ?d2)" 
            => "(sum (combine ?t0 ?d1 ?d2) ?d0)" 
            if is_unique(&["?d0", "?d1"]) ),

    ];

    let mut rules3 = vec![
        rw!("partitioned-matmul"; 
            "(matmul (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d2 ?d1 ?i0))"
             => "(partial_sum (matmul ?t0 ?t1) ?d1 ?i0)"
            if is_datadim(&["?d0"], vec!["data_dim0".to_string()])
            if is_datadim(&["?d2"], vec!["data_dim1".to_string()]) ),

        rw!("partitioned-matmul1"; 
            "(matmul (partition ?t0 ?d0 ?d1 ?i0) (replicate ?t1 ?d1 ?i0))" 
            => "(partition (matmul ?t0 ?t1) ?d0 ?d1 ?i0)" ),

        rw!("partitioned-matmul2"; 
            "(matmul (replicate ?t0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0))" 
            => "(partition (matmul ?t0 ?t1) ?d0 ?d1 ?i0)" ),

        rw!("partitioned-datadim2";
            "(matmul (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0))"
            => "(partition (matmul ?t0 ?t1) ?d0 ?d1 ?i0)"
            if is_datadim(&["?d0"], vec!["data_dim2".to_string()]) ),
    ];

   let mut rules4 = vec![
        rw!("reduce-partition-combine";
            "(partition (combine (partial_sum ?t0 ?d2 ?i0) ?d0 ?d2) ?d0 ?d1 ?i0)"
            => "(partial_sum ?t0 ?d1 ?i0)" ),

        rw!("reduce-partition-reduce-partition";
            "(reduce (partition (reduce (partition ?t0 ?d0 ?d1 ?i0) ?d1) ?d0 ?d2 ?i1) ?d2)"
            => "(reduce (partition ?t0 ?d0 ?d1 (dim_mul ?i0 ?i1)) ?d1)" ),

        rw!("reduce-reduce-partition-partition";
            "(reduce (reduce (partition (partition ?t0 ?d0 ?d1 ?i0) ?d0 ?d2 ?i1) ?d1) ?d2)"
            => "(reduce (partition ?t0 ?d0 ?d1 (dim_mul ?i0 ?i1)) ?d1)" ),

        rw!("reduce-reduce-partition-partition1";
            "(reduce (reduce (partition (partition ?t0 ?d0 ?d1 ?i0) ?d0 ?d2 ?i1) ?d2) ?d1)"
            => "(reduce (partition ?t0 ?d0 ?d1 (dim_mul ?i0 ?i1)) ?d1)" ),
    ]; 

    let mut rules5 = vec![

        rw!("sum-to-sumtox";
            "(sum (reduce (partition ?t0 ?d0 ?d1 ?i0) ?d1) ?d0)"
            => "(sum ?t0 ?d0)"),

        rw!("sum-to-comb-sum-par";
            "(sum (combine (sum (partition ?t0 ?d0 ?d2 ?i0) ?d0) ?d0 ?d2) ?d0)"
            => "(sum ?t0 ?d0)"),

        rw!("sum-to-sumtox-comb-par";
            "(sum (combine (reduce (partition (partition ?t0 ?d0 ?d2 ?i0) ?d0 ?d1 ?i1) ?d1) ?d0 ?d2) ?d0)"
            => "(sum ?t0 ?d0)" ),

        rw!("sum-to-sumtox-comb-par2";
            "(sum (combine (reduce (partition (partition ?t0 ?d0 ?d1 ?i0) ?d0 ?d2 ?i1) ?d1) ?d0 ?d2) ?d0)"
            => "(sum ?t0 ?d0)" ),

        rw!("sum-to-sumtox-comb-par3";
            "(sum (reduce (combine (partition (partition ?t0 ?d0 ?d2 ?i0) ?d0 ?d1 ?i1) ?d0 ?d2) ?d1) ?d0)"
            => "(sum ?t0 ?d0)" ),
    ];

    let mut rules6 = vec![
        rw!("ew-add-partition"; 
            "(combine (ew_add (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(ew_add ?t0 ?t1)" ),

        rw!("ew-mul-partition"; 
            "(combine (ew_mul (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(ew_mul ?t0 ?t1)" ),

        rw!("bc-div-partition"; 
            "(combine (bc_div (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(bc_div ?t0 ?t1)" ),

        rw!("bc-pow-partition"; 
            "(combine (bc_pow (partition ?t0 ?d0 ?d1 ?i0) (partition ?t1 ?d0 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(bc_pow ?t0 ?t1)" ),

        rw!("ew-add-replicate"; 
            "(combine (ew_add (partition ?t0 ?d0 ?d1 ?i0) (replicate ?t1 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(ew_add ?t0 ?t1)" ),

        rw!("ew-mul-replicate"; 
            "(combine (ew_mul (partition ?t0 ?d0 ?d1 ?i0) (replicate ?t1 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(ew_mul ?t0 ?t1)" ),

        rw!("bc-div-replicate"; 
            "(combine (bc_div (partition ?t0 ?d0 ?d1 ?i0) (replicate ?t1 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(bc_div ?t0 ?t1)" ),

        rw!("bc-pow-replicate"; 
            "(combine (bc_pow (partition ?t0 ?d0 ?d1 ?i0) (replicate ?t1 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(bc_pow ?t0 ?t1)" ),

        rw!("ew-add-replicate1"; 
            "(combine (ew_add (replicate ?t1 ?d1 ?i0) (partition ?t0 ?d0 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(ew_add ?t0 ?t1)" ),

        rw!("ew-mul-replicate1"; 
            "(combine (ew_mul (replicate ?t1 ?d1 ?i0) (partition ?t0 ?d0 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(ew_mul ?t0 ?t1)" ),

        rw!("bc-div-replicate1"; 
            "(combine (bc_div (replicate ?t1 ?d1 ?i0) (partition ?t0 ?d0 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(bc_div ?t0 ?t1)" ),

        rw!("bc-pow-replicate1";
            "(combine (bc_pow (replicate ?t1 ?d1 ?i0) (partition ?t0 ?d0 ?d1 ?i0)) ?d0 ?d1)"
            =>  "(bc_pow ?t0 ?t1)" ),

    ];

    let mut rules7 = vec![

        rw!("partition-exp"; 
            "(partition (ew_exp ?t0) ?d0 ?d1 ?i0)" 
            <=> "(ew_exp (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-square"; 
            "(partition (square ?t0) ?d0 ?d1 ?i0)" 
            <=> "(square (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-sqrt"; 
            "(partition (sqrt ?t0) ?d0 ?d1 ?i0)" 
            <=> "(sqrt (partition ?t0 ?d0 ?d1 ?i0))"),

        rw!("partition-silu"; 
            "(partition (silu ?t0) ?d0 ?d1 ?i0)" 
            <=> "(silu (partition ?t0 ?d0 ?d1 ?i0))" ),

        rw!("partition-gelu"; 
            "(partition (gelu ?t0) ?d0 ?d1 ?i0)" 
            <=> "(gelu (partition ?t0 ?d0 ?d1 ?i0))" ),

        rw!("partition-relu"; 
            "(partition (relu ?t0) ?d0 ?d1 ?i0)" 
            <=> "(relu (partition ?t0 ?d0 ?d1 ?i0))" ),

        rw!("partition-clamp"; 
            "(partition (clamp ?t0) ?d0 ?d1 ?i0)" 
            <=> "(clamp (partition ?t0 ?d0 ?d1 ?i0))" ),

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
            <=> "(silu (combine ?t0 ?d0 ?d1))" ),

        rw!("combine-gelu"; 
            "(combine (gelu ?t0) ?d0 ?d1)"
            <=> "(gelu (combine ?t0 ?d0 ?d1))"),

        rw!("combine-relu"; 
            "(combine (relu ?t0) ?d0 ?d1)" 
            <=> "(relu (combine ?t0 ?d0 ?d1))" ),

        rw!("combine-clamp"; 
            "(combine (clamp ?t0) ?d0 ?d1)" 
            <=> "(clamp (combine ?t0 ?d0 ?d1))" ),

        rw!("replicate-exp"; 
            "(replicate (ew_exp ?t0) ?d ?i0)" 
            <=> "(ew_exp (replicate ?t0 ?d ?i0))"),

        rw!("replicate-square"; 
            "(replicate (square ?t0) ?d ?i0)" 
            <=> "(square (replicate ?t0 ?d ?i0))"),

        rw!("replicate-sqrt"; 
            "(replicate (sqrt ?t0) ?d ?i0)" 
            <=> "(sqrt (replicate ?t0 ?d ?i0))"),

        rw!("replicate-silu"; 
            "(replicate (silu ?t0) ?d0 ?i0)" 
            <=> "(silu (replicate ?t0 ?d0 ?i0))" ),

        rw!("replicate-gelu"; 
            "(replicate (gelu ?t0) ?d0 ?i0)" 
            <=> "(gelu (replicate ?t0 ?d0 ?i0))" ),

        rw!("replicate-relu"; 
            "(replicate (relu ?t0) ?d0 ?i0)" 
            <=> "(relu (replicate ?t0 ?d0 ?i0))" ),

        rw!("replicate-clamp"; 
            "(replicate (clamp ?t0) ?d0 ?i0)" 
            <=> "(clamp (replicate ?t0 ?d0 ?i0))" ),

    ].concat();

    let mut rules8 = vec![

        rw!("partition-rms-norm"; 
            "(partition (rms_norm ?t0 ?d2) ?d0 ?d1 ?i0)" 
            <=> "(rms_norm (partition ?t0 ?d0 ?d1 ?i0) ?d2)"
            if is_unique(&["?d0", "?d1", "?d2"]) ),

        rw!("partition-rms";
            "(partition (rms ?t0 ?d2) ?d0 ?d1 ?i0)" 
            <=> "(rms (partition ?t0 ?d0 ?d1 ?i0) ?d2)"
            if is_unique(&["?d0", "?d1", "?d2"]) ),

        rw!("replicate-rms-norm";
            "(replicate (rms_norm ?t0 ?d1) ?d0 ?i0)" 
            <=> "(rms_norm (replicate ?t0 ?d0 ?i0) ?d1)" ),

        rw!("replicate-rms";
            "(replicate (rms ?t0 ?d2) ?d0 ?i0)" 
            <=> "(rms (replicate ?t0 ?d0 ?i0) ?d2)" ),

        rw!("combine-rms-norm";
            "(combine (rms_norm ?t0 ?d2) ?d0 ?d1)" 
            <=> "(rms_norm (combine ?t0 ?d0 ?d1) ?d2)"
            if is_unique(&["?d0", "?d1", "?d2"]) ),

        rw!("combine-rms";
            "(combine (rms ?t0 ?d2) ?d0 ?d1)" 
            <=> "(rms (combine ?t0 ?d0 ?d1) ?d2)"
            if is_unique(&["?d0", "?d1", "?d2"]) ),

    ].concat();

    let mut rules9 = vec![

        rw!("bc-div-commute-partition";
            "(bc_div (partition ?t0 ?d0 ?d1 ?i0) ?t1)"
            <=> "(partition (bc_div ?t0 ?t1) ?d0 ?d1 ?i0)" 
            if is_datadim(&["?d0"], vec!["data_dim0".to_string()]) ),

    ].concat();

    for i in 0..nums.len() {
        for j in (i+1)..nums.len() {
            let a = *(&mut nums[i]);
            let b = *(&mut nums[j]);
            let x = a * b;
            let base = format!("(dim_mul {} {})", a, b);
            let jk_expr = format!("{}", x);

            let rule_name = format!("factor-{}-{}", a, b);
            if !rules.iter().any(|r| r.name.as_str() == rule_name) {
                let lhs: Pattern<Expr> = base.parse().unwrap();
                let rhs: Pattern<Expr> = jk_expr.parse().unwrap();
                rules.push(Rewrite::new(rule_name, lhs, rhs).unwrap());
            }
        }
    }

    rules.append(&mut rules1);
    rules.append(&mut rules2);
    rules.append(&mut rules3);
    rules.append(&mut rules4);
    rules.append(&mut rules5);
    rules.append(&mut rules6);
    rules.append(&mut rules7);
    rules.append(&mut rules8);
    rules.append(&mut rules9);

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

    let re = Regex::new(r" (\d+)\)").unwrap();

    let mut nums: Vec<u32> = re.captures_iter(expr2_str)
        .filter_map(|cap| cap.get(1).unwrap().as_str().parse::<u32>().ok())
        .collect();

    let runner = Runner::default()
    .with_iter_limit(100_000)
    .with_node_limit(100_000)
    .with_time_limit(Duration::from_secs(1000))
    .with_expr(&expr2_str.parse().unwrap())
    .run(&rules(nums.clone()));

    let id1 = runner.egraph.equivs(&expr1_str.parse().unwrap(), &expr2_str.parse().unwrap());

    id1.len() > 0

}
