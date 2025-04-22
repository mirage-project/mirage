use std::ffi::{CStr};
use std::os::raw::c_char;

use egg::{rewrite as rw, *};
use ordered_float::NotNan;

pub type EGraph = egg::EGraph<Expr, ConstantFold>;
pub type Rewrite = egg::Rewrite<Expr, ConstantFold>;
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
        "factor" = Factor(Id),

        Constant(Constant),
        Symbol(Symbol),
    }
}


#[derive(Default)]
pub struct ConstantFold;
impl Analysis<Expr> for ConstantFold {
    type Data = Option<(Constant, PatternAst<Expr>)>;

    fn make(egraph: &mut EGraph, enode: &Expr) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0);
        Some(match enode {
            Expr::Constant(c) => (*c, format!("{}", c).parse().unwrap()),
            Expr::CMul([a, b]) => (
                x(a)? * x(b)?,
                format!("(cmul {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Expr::Factor(a) => (
                x(a)?,
                format!("(factor {})", x(a)?).parse().unwrap(),
            ),
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let data = egraph[id].data.clone();
        if let Some((c, pat)) = data {
            if egraph.are_explanations_enabled() {
                egraph.union_instantiations(
                    &pat,
                    &format!("{}", c).parse().unwrap(),
                    &Default::default(),
                    "constant_fold".to_string(),
                );
            } else {
                let added = egraph.add(Expr::Constant(c));
                egraph.union(id, added);
            }
            // to not prune, comment this out
            // egraph[id].nodes.retain(|n| n.is_leaf());

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}


fn is_not_zero(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        if let Some(n) = &egraph[subst[var]].data {
            *(n.0) != 0.0
        } else {
            true
        }
    }
}

fn is_const(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| egraph[subst[var]].data.is_some()
}

pub fn factor_rules(max: u32) -> Vec<Rewrite> {
    let mut rules = vec![];
    for i in 2..=max {
        for j in 2..=((i as f64).sqrt() as u32) {
        // for j in 2..=i - 1  {
            if i % j == 0 {
                let k = i / j;

                let name = format!("factor-{}-{}", k, j);
                let lhs: Pattern<Expr> = format!("(factor {})", i).parse().unwrap();
                let rhs: Pattern<Expr> = format!("(cmul {} {})", j, k).parse().unwrap();
                rules.push(Rewrite::new(name, lhs, rhs).unwrap());

                let name_sum = format!("sum-{}-{}", k, j);
                let lhs_sum: Pattern<Expr> = format!("(sum (cmul {} {}) ?a)", j, k).parse().unwrap();
                let rhs_sum: Pattern<Expr> = format!("(sum (factor {}) (sum (factor {}) ?a))", j, k).parse().unwrap();
                rules.push(Rewrite::new(name_sum, lhs_sum, rhs_sum).unwrap());
            }
        }
    }

    rules

}

#[rustfmt::skip]
pub fn rules() -> Vec<Rewrite> {
    
    let mut base_rules: Vec<Rewrite> = vec![
        rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
        rw!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
        rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rw!("assoc-add-inv"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rw!("assoc-mul-inv"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),

        rw!("distribute"; "(* ?a (+ ?b ?c))"  => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("distribute-inv"; "(+ (* ?a ?b) (* ?a ?c))"=> "(* ?a (+ ?b ?c))"),

        rw!("div-div"; "(+ (/ ?a ?b) (/ ?c ?b))" => "(/ (+ ?a ?c) ?b)" if is_not_zero("?b")),
        rw!("div-div-inv"; "(/ (+ ?a ?c) ?b)" => "(+ (/ ?a ?b) (/ ?c ?b))" if is_not_zero("?b")),
        rw!("mul-div"; "(* ?a (/ ?b ?c))" => "(/ (* ?a ?b) ?c)" if is_not_zero("?c")),
        rw!("mul-div-inv"; "(/ (* ?a ?b) ?c)" => "(* ?a (/ ?b ?c))" if is_not_zero("?c")),
        rw!("div-div"; "(/ (/ ?a ?b) ?c)" => "(/ ?a (* ?b ?c))" if is_not_zero("?b") if is_not_zero("?c")),
        rw!("div-div-inv"; "(/ ?a (* ?b ?c))" => "(/ (/ ?a ?b) ?c)" if is_not_zero("?b") if is_not_zero("?c")),

        rw!("sum-one"; "(sum (factor 1) ?a)" => "(?a)"),
        rw!("assoc-sum"; "(sum ?i (sum ?j ?a))" => "(sum (cmul ?i ?j) ?a)" if is_const("?i") if is_const("?j")),
        rw!("comm-sum"; "(sum (factor ?i) (sum (factor ?j) ?a))" => "(sum (factor ?j) (sum (factor ?i) ?a))" if is_const("?i") if is_const("?j")),
        rw!("sum-add"; "(sum (factor ?i) (+ ?a ?b))" => "(+ (sum (factor ?i) ?a) (sum (factor ?i) ?b))" if is_const("?i")),
        rw!("sum-add-inv"; "(+ (sum (factor ?i) ?a) (sum (factor ?i) ?b))" => "(sum (factor ?i) (+ ?a ?b))" if is_const("?i")),
        rw!("sum-mul"; "(sum (factor ?i) (* ?a ?b))" => "(* (sum (factor ?i) ?a) ?b)" if is_const("?i")),
        rw!("sum-mul-inv"; "(* (sum (factor ?i) ?a) ?b)" => "(sum (factor ?i) (* ?a ?b))" if is_const("?i")),
        rw!("sum-div"; "(sum (factor ?i) (/ ?a ?b))" => "(/ (sum (factor ?i) ?a) ?b)" if is_const("?i")), 
        rw!("sum-div-inv"; "(/ (sum (factor ?i) ?a) ?b)" => "(sum (factor ?i) (/ ?a ?b))" if is_const("?i")),
    ];

    base_rules.extend(factor_rules(4096));

    base_rules
}

#[unsafe(no_mangle)]
pub extern "C" fn egg_equiv(expr1: *const c_char, expr2: *const c_char) -> bool {
    let expr1_str = unsafe { CStr::from_ptr(expr1) }.to_str().unwrap_or("");
    let expr2_str = unsafe { CStr::from_ptr(expr2) }.to_str().unwrap_or("");

    let mut egraph :egg::EGraph<Expr, ConstantFold> = Default::default();
    egraph.add_expr(&expr1_str.parse().unwrap());
    let egraph = Runner::default()
        .with_egraph(egraph)
        .run(&rules())
        .with_iter_limit(100000)
        .egraph;

    let subexpr: RecExpr<Expr> = expr2_str.parse().unwrap();
    let sub_id = egraph.lookup_expr(&subexpr);

    sub_id.is_some()
}