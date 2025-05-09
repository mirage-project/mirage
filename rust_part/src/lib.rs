use egg::{rewrite as rw, *};
use ordered_float::NotNan;
use std::{time::Duration};
use std::ffi::CStr;
use regex::Regex;
use std::os::raw::c_char;

pub type EGraph = egg::EGraph<Expr, ()>;
pub type Rewrite = egg::Rewrite<Expr, ()>;
pub type Constant = NotNan<f64>;

define_language! {
    pub enum Expr {
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = SMul([Id; 2]),
        "/" = Div([Id; 2]),
        "cmul" = CMul([Id; 2]),

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
pub fn rules(max: u32) -> Vec<Rewrite> {
    let mut rules = vec![
        rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
        rw!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
        rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rw!("assoc-add-inv"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rw!("assoc-mul-inv"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        rw!("comm-cmul"; "(cmul ?a ?b)" => "(cmul ?b ?a)"),

        rw!("distribute"; "(* ?a (+ ?b ?c))"  => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("distribute-inv"; "(+ (* ?a ?b) (* ?a ?c))"=> "(* ?a (+ ?b ?c))"),

        rw!("div-add"; "(+ (/ ?a ?b) (/ ?c ?b))" => "(/ (+ ?a ?c) ?b)" ),
        rw!("div-add-inv"; "(/ (+ ?a ?c) ?b)" => "(+ (/ ?a ?b) (/ ?c ?b))"),
        rw!("mul-div"; "(* ?a (/ ?b ?c))" => "(/ (* ?a ?b) ?c)" ),
        rw!("mul-div-inv"; "(/ (* ?a ?b) ?c)" => "(* ?a (/ ?b ?c))" ),
        rw!("div-div"; "(/ (/ ?a ?b) ?c)" => "(/ ?a (* ?b ?c))" ),
        rw!("div-div-inv"; "(/ ?a (* ?b ?c))" => "(/ (/ ?a ?b) ?c)" ),

        rw!("sum-one"; "(sum 1 ?a)" => "(?a)"),
        rw!("assoc-sum"; "(sum ?i (sum ?j ?a))" => "(sum (cmul ?i ?j) ?a)" ),
        rw!("assoc-sum-inv"; "(sum (cmul ?i ?j) ?a)" => "(sum ?i (sum ?j ?a))" ),
        rw!("comm-sum"; "(sum ?i (sum ?j ?a))" => "(sum ?j (sum ?i ?a))" ),
        rw!("sum-add"; "(sum ?i (+ ?a ?b))" => "(+ (sum ?i ?a) (sum ?i ?b))"),
        rw!("sum-add-inv"; "(+ (sum ?i ?a) (sum ?i ?b))" => "(sum ?i (+ ?a ?b))" ),
        rw!("sum-mul"; "(sum ?i (* ?a ?b))" => "(* (sum ?i ?a) ?b)" ),
        rw!("sum-mul-2"; "(sum ?i (* ?a ?b))" => "(* (sum ?i ?b) ?a)" ),
        rw!("sum-mul-inv"; "(* (sum ?i ?a) ?b)" => "(sum ?i (* ?a ?b))" ),
        rw!("sum-div"; "(sum ?i (/ ?a ?b))" => "(/ (sum ?i ?a) ?b)" ), 
        rw!("sum-div-inv"; "(/ (sum ?i ?a) ?b)" => "(sum ?i (/ ?a ?b))"),

    ];

    for i in 2..=max {
        for j in 2..=((i as f64).sqrt() as u32) {
        // for j in 2..=i - 1  {
            if i % j == 0 {
                let k = i / j;

                let name = format!("factor-{}-{}", k, j);
                let lhs: Pattern<Expr> = format!("{}", i).parse().unwrap();
                let rhs: Pattern<Expr> = format!("(cmul {} {})", j, k).parse().unwrap();
                rules.push(Rewrite::new(name, lhs, rhs).unwrap());
                let name_inv = format!("factor-{}-{}-inv", k, j);
                let lhs_inv: Pattern<Expr> = format!("(cmul {} {})", j, k).parse().unwrap();
                let rhs_inv: Pattern<Expr> = format!("{}", i).parse().unwrap();
                rules.push(Rewrite::new(name_inv, lhs_inv, rhs_inv).unwrap());
            }
        }
    }
    rules
}

#[no_mangle]
pub extern "C" fn egg_equiv(expr1: *const c_char, expr2: *const c_char) -> bool {
    let expr1_str: &str = unsafe {
            CStr::from_ptr(expr1)
        }.to_str().unwrap_or("");

    let re = Regex::new(r"\(sum (\d+)").unwrap();

    let nums: Vec<i32> = re.captures_iter(expr1_str)
        .filter_map(|cap| cap.get(1).unwrap().as_str().parse::<i32>().ok())
        .collect();
    
    let max_number_1 = if nums.is_empty() {
        2
    } else {
        nums.iter().product()
    };

    let expr2_str: &str = unsafe {
        CStr::from_ptr(expr2)
    }.to_str().unwrap_or("");
    let nums: Vec<i32> = re.captures_iter(expr2_str)
        .filter_map(|cap| cap.get(1).unwrap().as_str().parse::<i32>().ok())
        .collect();
    
    let max_number_2 = if nums.is_empty() {
        2
    } else {
        nums.iter().product()
    };
    let max_number = max_number_1.max(max_number_2);
    let max_number = max_number as u32;
    
    let mut egraph :egg::EGraph<Expr, ()> = Default::default();
    egraph.add_expr(&expr2_str.parse().unwrap());
    let duration = Duration::from_secs(400);
    let runner: Runner<Expr, ()> = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(100_000)
        .with_node_limit(10_000)
        .with_time_limit(duration)
        .run(&rules(max_number));

    let subexpr: RecExpr<Expr> = expr1_str.parse().unwrap();
    let sub_id = runner.egraph.lookup_expr(&subexpr);

    sub_id.is_some()
}