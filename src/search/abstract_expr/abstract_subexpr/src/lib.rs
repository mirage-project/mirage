use egg::{rewrite as rw, *};
use std::{time::Duration};
use std::ffi::CStr;
use regex::Regex;
use std::os::raw::{c_char, c_int};

pub type EGraph = egg::EGraph<Expr, ()>;
pub type Rewrite = egg::Rewrite<Expr, ()>;
pub type Constant = i32;

define_language! {
    pub enum Expr {
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = SMul([Id; 2]),
        "/" = Div([Id; 2]),

        "sum" = Sum([Id; 2]),
        "exp" = Exp(Id),
        "pow" = Pow([Id; 2]),
        "square" = Square(Id),
        "sqrt" = Sqrt(Id),
        "silu" = Silu(Id),
        "gelu" = Gelu(Id),
        "relu" = Relu(Id),
        "clamp" = Clamp([Id; 3]),
        "rms" = Rms([Id; 2]),

        Constant(Constant),
        Symbol(Symbol),
    }
}


#[rustfmt::skip]
pub fn rules(mut nums: Vec<u32>) -> Vec<Rewrite> {
    let mut rules = vec![
        rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
        rw!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
        rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rw!("assoc-add-inv"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rw!("assoc-mul-inv"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),

        rw!("distribute"; "(* ?a (+ ?b ?c))"  => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("distribute-inv"; "(+ (* ?a ?b) (* ?a ?c))"=> "(* ?a (+ ?b ?c))"),

        rw!("div-add"; "(+ (/ ?a ?b) (/ ?c ?b))" => "(/ (+ ?a ?c) ?b)" ),
        rw!("div-add-inv"; "(/ (+ ?a ?c) ?b)" => "(+ (/ ?a ?b) (/ ?c ?b))"),
        rw!("mul-div"; "(* ?a (/ ?b ?c))" => "(/ (* ?a ?b) ?c)" ),
        rw!("mul-div-inv"; "(/ (* ?a ?b) ?c)" => "(* ?a (/ ?b ?c))" ),
        rw!("div-div"; "(/ (/ ?a ?b) ?c)" => "(/ ?a (* ?b ?c))" ),
        rw!("div-div-inv"; "(/ ?a (* ?b ?c))" => "(/ (/ ?a ?b) ?c)" ),

        rw!("comm-sum"; "(sum ?i (sum ?j ?a))" => "(sum ?j (sum ?i ?a))" ),
        rw!("sum-add"; "(sum ?i (+ ?a ?b))" => "(+ (sum ?i ?a) (sum ?i ?b))"),
        rw!("sum-add-inv"; "(+ (sum ?i ?a) (sum ?i ?b))" => "(sum ?i (+ ?a ?b))" ),
        rw!("sum-mul"; "(sum ?i (* ?a ?b))" => "(* (sum ?i ?a) ?b)" ),
        rw!("sum-mul-2"; "(sum ?i (* ?a ?b))" => "(* (sum ?i ?b) ?a)" ),
        rw!("sum-mul-inv"; "(* (sum ?i ?a) ?b)" => "(sum ?i (* ?a ?b))" ),
        rw!("sum-div"; "(sum ?i (/ ?a ?b))" => "(/ (sum ?i ?a) ?b)" ), 
        rw!("sum-div-inv"; "(/ (sum ?i ?a) ?b)" => "(sum ?i (/ ?a ?b))"),

    ];
    let sum_rules = vec![
        rw!("sum-mul-mul"; "(sum ?i (sum ?j (* ?a ?b)))" => "(* (sum ?i ?a) (sum ?j ?b))"),
        rw!("sum-mul-mul-inv"; "(* (sum ?i ?a) (sum ?j ?b))" => "(sum ?i (sum ?j (* ?a ?b)))"),
    ];

    let mut extra_rules: Vec<Rewrite> = Vec::new();

    nums.sort();
    nums.dedup();
    // EVERY factor pair of n, not just repeated division by the smallest prime.
    // The old loop shrank `n` inside the inner loop, so `j * j <= n` collapsed
    // and it only ever peeled the smallest prime off the top: for 3072 = 2^10*3
    // it emitted (sum 3072)=(sum 2 (sum 1536)) ... down to (sum 6)=(sum 2 (sum 3)),
    // but never a pair carrying the 3 at the top, such as (sum 3072)=(sum 24
    // (sum 128)). A tiled matmul's abstract expression is (sum forloop_range
    // (sum k_tile ...)), so without that pair it can never be shown equal to
    // the spec's (sum K ...) -- the candidate is pruned and search returns only
    // the trivial kernel-level graph. Powers of two were unaffected because
    // every factor pair of 2^a is reachable by repeated halving.
    for &mut num in &mut nums {
        let n = num;
        let mut j = 2;
        while j * j <= n {
            if n % j == 0 {
                for (a, b) in [(j, n / j), (n / j, j)] {
                    if b == 1 {
                        continue;
                    }
                    let base = format!("(sum {} ?a)", n);
                    let ab_expr = format!("(sum {} (sum {} ?a))", a, b);
                    let rule_name = format!("factor-{}-{}-{}", n, a, b);
                    if !extra_rules.iter().any(|r| r.name.as_str() == rule_name) {
                        let lhs: Pattern<Expr> = base.parse().unwrap();
                        let rhs: Pattern<Expr> = ab_expr.parse().unwrap();
                        extra_rules.push(Rewrite::new(rule_name, lhs, rhs).unwrap());
                    }
                    let rule_name_inv = format!("factor-{}-{}-{}-inv", n, a, b);
                    if !extra_rules.iter().any(|r| r.name.as_str() == rule_name_inv) {
                        let lhs_inv: Pattern<Expr> = ab_expr.parse().unwrap();
                        let rhs_inv: Pattern<Expr> = base.parse().unwrap();
                        extra_rules.push(Rewrite::new(rule_name_inv, lhs_inv, rhs_inv).unwrap());
                    }
                }
            }
            j = j + 1;
        }
    }
    rules.extend(sum_rules);
    rules.extend(extra_rules);

    rules
}


#[repr(C)]
pub struct KVPair {
    key: i32,
    value: bool,
}

static mut Graphs: Vec<EGraph> = Vec::new();

#[no_mangle]
pub extern "C" fn get_egraph(expr: *const c_char) -> () {
    let expr_str: &str = unsafe {
            CStr::from_ptr(expr)
        }.to_str().unwrap_or("");

    let re = Regex::new(r"\(sum (\d+)\b").unwrap();

    let nums: Vec<u32> = re.captures_iter(expr_str)
        .filter_map(|cap| cap.get(1).unwrap().as_str().parse::<u32>().ok())
        .collect();

    let duration = Duration::from_secs(10);

    let runner: Runner<Expr, ()> = Runner::default()
        .with_iter_limit(100_000)
        .with_node_limit(100_000)
        .with_time_limit(duration)
        .with_expr(&expr_str.parse().unwrap())
        .run(&rules(nums));

    unsafe{
        Graphs.push(runner.egraph);   
    }

}

#[no_mangle]
pub extern "C" fn egg_equiv(subexprs: *const *const c_char, len: c_int) -> *mut bool {
    
    let subexpr_vec: Vec<String> = unsafe {
        (0..len)
            .map(|i| {
                let cstr_ptr = *subexprs.add(i as usize); 
                if cstr_ptr.is_null() {
                    "__PLACEHOLDER_NULL__".to_string()
                } else {
                    CStr::from_ptr(cstr_ptr).to_string_lossy().into_owned() 
                }
            })
            .collect()
    };    
    
    let mut data: Vec<bool> = Vec::new();
    for i in 0..len {
        let subexpr_str = &subexpr_vec[i as usize];
        let mut result: bool = false;
        if subexpr_str != "null" {
            let subexpr: RecExpr<Expr> = subexpr_str.parse().unwrap();
            for graph in unsafe { &Graphs } {
                let sub_id = graph.lookup_expr(&subexpr);
                if sub_id.is_some() {
                    result = true;
                    break;
                }
            }
        }
        data.push(result);
    }
    let boxed = data.into_boxed_slice();

    Box::into_raw(boxed) as *mut bool
}