library("opencesp")
library("Rcpp")
library("treeClust")
library("ggplot2")
library("janitor")
library("AppliedPredictiveModeling")
library("randomForestSRC")
library("randomForest")
library("genridge")
library("medicaldata")
library("patchwork")
library("grid")
library("parallel")
library("future")
library("future.apply")
library("ggpattern")
library("randomForest")

save_dir <- ""
wd <- ""

bor_cols <- function() {
  c("#534785", "#f55f5e", "#dec91a", "#51A3A3", "#82846D", "#A1C084", "#754F5B","#27476E", "#EB9486")
}

save_plot <- function(dir, width=7, height=5.5, p = NULL) {
  files <- list.files(path = dir, pattern = "plot[0-9]+\\.jpg", full.names = FALSE)
  n <- as.numeric(gsub("plot([0-9]+)\\.jpg", "\\1", files))
  if (length(n) > 0) max_n <- max(n, na.rm = TRUE)
  else max_n <- 0
  if(is.null(p)) p <- ggplot2::last_plot()
  filename <- paste(dir, "/plot", max_n + 1, ".jpg", sep="")
  ggplot2::ggsave(file = filename, plot = p,
         width = width, height = height, dpi = 320)
  return(filename)
}

tree_engine_includes <- '
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

using namespace Rcpp;
using namespace std;

struct Split {
  bool ok;
  int type;
  int feature;
  float threshold;
  float gain;
  vector<int> cats;

  Split() : ok(false), type(0), feature(-1), threshold(0.0f), gain(0.0f) {}
};

struct Node {
  int type;
  int feature;
  float threshold;
  int left;
  int right;
  vector<int> cats;
  vector<int> rows;

  Node() : type(0), feature(-1), threshold(0.0f), left(-1), right(-1) {}
};

struct Tree {
  vector<Node> nodes;
};

struct Forest {
  vector<Tree> trees;
};

struct Data {
  int n;
  int p_num;
  int p_cat;
  bool y_factor;
  int y_levels;
  vector<float> x_num;
  vector<int> x_cat;
  vector<float> y;

  Data(NumericMatrix x_num_, IntegerMatrix x_cat_, NumericVector y_, bool y_factor_, int y_levels_) {
    n = y_.size();
    p_num = x_num_.ncol();
    p_cat = x_cat_.ncol();
    y_factor = y_factor_;
    y_levels = y_levels_;

    x_num.resize(n * p_num);
    x_cat.resize(n * p_cat);
    y.resize(n);

    for(int j = 0; j < p_num; ++j) {
      for(int i = 0; i < n; ++i) {
        x_num[i + n * j] = (float) x_num_(i, j);
      }
    }

    for(int j = 0; j < p_cat; ++j) {
      for(int i = 0; i < n; ++i) {
        x_cat[i + n * j] = x_cat_(i, j);
      }
    }

    for(int i = 0; i < n; ++i) {
      y[i] = (float) y_[i];
    }
  }

  inline float num(int i, int j) const {
    return x_num[i + n * j];
  }

  inline int cat(int i, int j) const {
    return x_cat[i + n * j];
  }

  inline int cls(int i) const {
    return (int) y[i];
  }
};

inline int rnd(int n) {
  return (int) (R::unif_rand() * n);
}

inline float rss(float s, float ss, int n) {
  return ss - s * s / n;
}

inline float gini(const vector<int>& count, int n) {
  float q = 0.0f;

  for(int k = 1; k < (int) count.size(); ++k) {
    q += (float) count[k] * count[k];
  }

  return n - q / n;
}

inline float gini_split(const vector<int>& total, const vector<int>& left, int left_n, int n) {
  int right_n = n - left_n;
  float left_q = 0.0f;
  float right_q = 0.0f;

  for(int k = 1; k < (int) total.size(); ++k) {
    int right = total[k] - left[k];
    left_q += (float) left[k] * left[k];
    right_q += (float) right * right;
  }

  return left_n - left_q / left_n + right_n - right_q / right_n;
}

struct Grower {
  const Data& data;
  int min_bucket;
  Tree tree;

  Grower(const Data& data_, int min_bucket_) : data(data_), min_bucket(min_bucket_) {}

  void leaf(int node, const vector<int>& rows) {
    tree.nodes[node].rows.resize(rows.size());

    for(int i = 0; i < (int) rows.size(); ++i) {
      tree.nodes[node].rows[i] = rows[i] + 1;
    }
  }

  Split split_reg(const vector<int>& rows) {
    Split best;
    int n = rows.size();

    float total_s = 0.0f;
    float total_ss = 0.0f;

    for(int id : rows) {
      float y = data.y[id];
      total_s += y;
      total_ss += y * y;
    }

    float parent = rss(total_s, total_ss, n);

    vector< pair<float, int> > values;
    values.reserve(n);

    for(int j = 0; j < data.p_num; ++j) {
      values.clear();

      for(int id : rows) {
        values.push_back(make_pair(data.num(id, j), id));
      }

      sort(values.begin(), values.end(), [](const pair<float, int>& a, const pair<float, int>& b) {
        return a.first == b.first ? a.second < b.second : a.first < b.first;
      });

      float left_s = 0.0f;
      float left_ss = 0.0f;

      for(int i = 0; i < n - 1; ++i) {
        int id = values[i].second;
        float y = data.y[id];

        left_s += y;
        left_ss += y * y;

        if(values[i].first == values[i + 1].first) continue;

        int left_n = i + 1;
        int right_n = n - left_n;

        if(left_n < min_bucket || right_n < min_bucket) continue;

        float right_s = total_s - left_s;
        float right_ss = total_ss - left_ss;
        float loss = rss(left_s, left_ss, left_n) + rss(right_s, right_ss, right_n);
        float gain = parent - loss;

        if(gain >= best.gain) {
          best.ok = true;
          best.type = 1;
          best.feature = j;
          best.threshold = values[i].first + 0.5f * (values[i + 1].first - values[i].first);
          best.gain = gain;
          best.cats.clear();
        }
      }
    }

    for(int j = 0; j < data.p_cat; ++j) {
      int max_cat = 0;

      for(int id : rows) {
        int value = data.cat(id, j);
        if(value > max_cat) max_cat = value;
      }

      vector<int> count(max_cat + 1, 0);
      vector<float> sum(max_cat + 1, 0.0f);
      vector<float> sumsq(max_cat + 1, 0.0f);

      for(int id : rows) {
        int value = data.cat(id, j);
        float y = data.y[id];

        count[value] += 1;
        sum[value] += y;
        sumsq[value] += y * y;
      }

      vector< pair<float, int> > order;

      for(int value = 1; value <= max_cat; ++value) {
        if(count[value]) {
          order.push_back(make_pair(sum[value] / count[value], value));
        }
      }

      sort(order.begin(), order.end(), [](const pair<float, int>& a, const pair<float, int>& b) {
        return a.first == b.first ? a.second < b.second : a.first < b.first;
      });

      int left_n = 0;
      float left_s = 0.0f;
      float left_ss = 0.0f;
      vector<int> left_cats;

      for(int pos = 0; pos + 1 < (int) order.size(); ++pos) {
        int value = order[pos].second;

        left_cats.push_back(value);
        left_n += count[value];
        left_s += sum[value];
        left_ss += sumsq[value];

        int right_n = n - left_n;

        if(left_n < min_bucket || right_n < min_bucket) continue;

        float right_s = total_s - left_s;
        float right_ss = total_ss - left_ss;
        float loss = rss(left_s, left_ss, left_n) + rss(right_s, right_ss, right_n);
        float gain = parent - loss;

        if(gain >= best.gain) {
          best.ok = true;
          best.type = 2;
          best.feature = j;
          best.threshold = 0.0f;
          best.gain = gain;
          best.cats = left_cats;
          sort(best.cats.begin(), best.cats.end());
        }
      }
    }

    return best;
  }

  Split split_cls(const vector<int>& rows) {
    Split best;
    int n = rows.size();

    vector<int> total(data.y_levels + 1, 0);

    for(int id : rows) {
      total[data.cls(id)] += 1;
    }

    float parent = gini(total, n);

    vector< pair<float, int> > values;
    values.reserve(n);

    for(int j = 0; j < data.p_num; ++j) {
      values.clear();

      for(int id : rows) {
        values.push_back(make_pair(data.num(id, j), id));
      }

      sort(values.begin(), values.end(), [](const pair<float, int>& a, const pair<float, int>& b) {
        return a.first == b.first ? a.second < b.second : a.first < b.first;
      });

      vector<int> left(data.y_levels + 1, 0);

      for(int i = 0; i < n - 1; ++i) {
        int id = values[i].second;

        left[data.cls(id)] += 1;

        if(values[i].first == values[i + 1].first) continue;

        int left_n = i + 1;
        int right_n = n - left_n;

        if(left_n < min_bucket || right_n < min_bucket) continue;

        float loss = gini_split(total, left, left_n, n);
        float gain = parent - loss;

        if(gain >= best.gain) {
          best.ok = true;
          best.type = 1;
          best.feature = j;
          best.threshold = values[i].first + 0.5f * (values[i + 1].first - values[i].first);
          best.gain = gain;
          best.cats.clear();
        }
      }
    }

    for(int j = 0; j < data.p_cat; ++j) {
      int max_cat = 0;

      for(int id : rows) {
        int value = data.cat(id, j);
        if(value > max_cat) max_cat = value;
      }

      int stride = data.y_levels + 1;
      vector<int> count(max_cat + 1, 0);
      vector<int> class_count((max_cat + 1) * stride, 0);

      for(int id : rows) {
        int value = data.cat(id, j);
        int cls = data.cls(id);

        count[value] += 1;
        class_count[value * stride + cls] += 1;
      }

      vector<int> cats;

      for(int value = 1; value <= max_cat; ++value) {
        if(count[value]) {
          cats.push_back(value);
        }
      }

      if((int) cats.size() < 2) continue;

      if(data.y_levels == 2) {
        vector< pair<float, int> > order;

        for(int value : cats) {
          float score = (float) class_count[value * stride + 2] / count[value];
          order.push_back(make_pair(score, value));
        }

        sort(order.begin(), order.end(), [](const pair<float, int>& a, const pair<float, int>& b) {
          return a.first == b.first ? a.second < b.second : a.first < b.first;
        });

        int left_n = 0;
        vector<int> left(data.y_levels + 1, 0);
        vector<int> left_cats;

        for(int pos = 0; pos + 1 < (int) order.size(); ++pos) {
          int value = order[pos].second;

          left_cats.push_back(value);
          left_n += count[value];

          for(int k = 1; k <= data.y_levels; ++k) {
            left[k] += class_count[value * stride + k];
          }

          int right_n = n - left_n;

          if(left_n < min_bucket || right_n < min_bucket) continue;

          float loss = gini_split(total, left, left_n, n);
          float gain = parent - loss;

          if(gain > best.gain) {
            best.ok = true;
            best.type = 2;
            best.feature = j;
            best.threshold = 0.0f;
            best.gain = gain;
            best.cats = left_cats;
            sort(best.cats.begin(), best.cats.end());
          }
        }
      }
      else {
        int q = cats.size();

        unsigned long long total_masks = 1ULL << (q - 1);

        for(unsigned long long mask = 1; mask < total_masks; ++mask) {
          int left_n = 0;
          vector<int> left(data.y_levels + 1, 0);
          vector<int> left_cats;

          for(int pos = 0; pos < q - 1; ++pos) {
            if(mask & (1ULL << pos)) {
              int value = cats[pos];

              left_cats.push_back(value);
              left_n += count[value];

              for(int k = 1; k <= data.y_levels; ++k) {
                left[k] += class_count[value * stride + k];
              }
            }
          }

          int right_n = n - left_n;

          if(left_n < min_bucket || right_n < min_bucket) continue;

          float loss = gini_split(total, left, left_n, n);
          float gain = parent - loss;

          if(gain > best.gain) {
            best.ok = true;
            best.type = 2;
            best.feature = j;
            best.threshold = 0.0f;
            best.gain = gain;
            best.cats = left_cats;
            sort(best.cats.begin(), best.cats.end());
          }
        }
      }
    }

    return best;
  }

  bool left_side(int id, const Split& split) {
    if(split.type == 1) {
      return data.num(id, split.feature) <= split.threshold;
    }

    return binary_search(split.cats.begin(), split.cats.end(), data.cat(id, split.feature));
  }

  int grow(const vector<int>& rows) {
    int node = tree.nodes.size();

    tree.nodes.push_back(Node());

    if(rows.size() < 2) {
      leaf(node, rows);
      return node;
    }

    Split split = data.y_factor ? split_cls(rows) : split_reg(rows);

    if(!split.ok) {
      leaf(node, rows);
      return node;
    }

    vector<int> left_rows;
    vector<int> right_rows;

    left_rows.reserve(rows.size());
    right_rows.reserve(rows.size());

    for(int id : rows) {
      if(left_side(id, split)) {
        left_rows.push_back(id);
      }
      else {
        right_rows.push_back(id);
      }
    }

    tree.nodes[node].type = split.type;
    tree.nodes[node].feature = split.feature;
    tree.nodes[node].threshold = split.threshold;
    tree.nodes[node].cats = split.cats;
    tree.nodes[node].left = grow(left_rows);
    tree.nodes[node].right = grow(right_rows);

    return node;
  }
};

int draw_one(const Tree& tree, NumericMatrix x_num, IntegerMatrix x_cat, int row) {
  int node = 0;

  while(tree.nodes[node].type) {
    const Node& nd = tree.nodes[node];

    if(nd.type == 1) {
      node = ((float) x_num(row, nd.feature) <= nd.threshold) ? nd.left : nd.right;
    }
    else {
      node = binary_search(nd.cats.begin(), nd.cats.end(), x_cat(row, nd.feature)) ? nd.left : nd.right;
    }
  }

  const vector<int>& rows = tree.nodes[node].rows;

  return rows[rnd(rows.size())];
}
'

tree_engine_code <- '
SEXP tree_engine(std::string op, List args) {
  RNGScope scope;

  if(op == "fit") {
    NumericMatrix x_num = args["x_num"];
    IntegerMatrix x_cat = args["x_cat"];
    NumericVector y = args["y"];

    Data data(
      x_num,
      x_cat,
      y,
      as<bool>(args["y_factor"]),
      as<int>(args["y_levels"])
    );

    bool bag = as<bool>(args["bag"]);
    float prop = as<float>(args["prop"]);
    int ntrees = as<int>(args["ntrees"]);
    int min_bucket = as<int>(args["min_bucket"]);
    int n = y.size();

    Forest* forest = new Forest();
    vector<int> full(n);

    for(int i = 0; i < n; ++i) {
      full[i] = i;
    }

    if(!bag) {
      Grower grower(data, min_bucket);
      grower.grow(full);
      forest->trees.push_back(grower.tree);
    }
    else {
      int m = (int) ceil(prop * n);
      forest->trees.reserve(ntrees);

      for(int tree_id = 0; tree_id < ntrees; ++tree_id) {
        vector<int> rows = full;

        for(int i = 0; i < m; ++i) {
          int j = i + rnd(n - i);
          swap(rows[i], rows[j]);
        }

        rows.resize(m);

        Grower grower(data, min_bucket);
        grower.grow(rows);
        forest->trees.push_back(grower.tree);
      }
    }

    XPtr<Forest> ptr(forest, true);
    return ptr;
  }

  SEXP ptr = args["model"];
  XPtr<Forest> forest(ptr);
  NumericMatrix x_num = args["x_num"];
  IntegerMatrix x_cat = args["x_cat"];
  int n = x_num.nrow();
  int ntrees = forest->trees.size();
  IntegerVector out(n);

  if(ntrees == 1) {
    const Tree& tree = forest->trees[0];

    for(int i = 0; i < n; ++i) {
      out[i] = draw_one(tree, x_num, x_cat, i);
    }
  }
  else {
    for(int i = 0; i < n; ++i) {
      out[i] = draw_one(forest->trees[rnd(ntrees)], x_num, x_cat, i);
    }
  }

  return out;
}
'

compile_tree_engine <- function() {
  e <- new.env(parent = .GlobalEnv)

  cppFunction(
    plugins = "cpp11",
    includes = tree_engine_includes,
    code = tree_engine_code,
    env = e
  )
  assign("tree_engine", get("tree_engine", envir = e, inherits = FALSE), envir = .GlobalEnv)
  invisible(NULL)
}
compile_tree_engine()

call_tree_engine <- function(op, args) {
  if(!exists("tree_engine", envir = .GlobalEnv, inherits = FALSE)) {
    compile_tree_engine()
  }

  get("tree_engine", envir = .GlobalEnv, inherits = FALSE)(op, args)
}

fit_cond <- function(data, target, predictors, minbucket = NULL, prop = NULL, ntrees = 1) {
  if(!length(predictors)) return(list(type = "marginal", target = target))

  is_num <- vapply(data[predictors], is.numeric, logical(1))
  num_predictors <- predictors[is_num]
  cat_predictors <- predictors[!is_num]

  if(length(num_predictors)) {
    x_num <- as.matrix(data[num_predictors])
    storage.mode(x_num) <- "double"
  }
  else x_num <- matrix(numeric(0), nrow = nrow(data), ncol = 0)

  if(length(cat_predictors)) {
    x_cat <- matrix(0, nrow = nrow(data), ncol = length(cat_predictors))

    for(j in seq_along(cat_predictors)) {
      x_cat[, j] <- as.integer(data[[cat_predictors[j]]])
    }
  }
  else x_cat <- matrix(integer(0), nrow = nrow(data), ncol = 0)

  y_factor <- is.factor(data[[target]])

  model <- call_tree_engine("fit", list(
    x_num = x_num,
    x_cat = x_cat,
    y = as.numeric(data[[target]]),
    y_factor = y_factor,
    y_levels = if(y_factor) nlevels(data[[target]]) else 0,
    bag = ntrees > 1,
    prop = if(ntrees > 1) as.numeric(prop) else 1,
    ntrees = as.integer(ntrees),
    min_bucket = as.integer(minbucket)
  ))

  list(
    type = if(ntrees > 1) "subagging" else "tree",
    target = target,
    num_predictors = num_predictors,
    cat_predictors = cat_predictors,
    model = model
  )
}

draw_cond <- function(model, data, synth) {
  if(model$type == "marginal") return(data[[model$target]][sample.int(nrow(data), nrow(synth), replace = TRUE)])

  if(length(model$num_predictors)) {
    x_num <- as.matrix(synth[model$num_predictors])
    storage.mode(x_num) <- "double"
  }
  else x_num <- matrix(numeric(0), nrow = nrow(synth), ncol = 0)

  if(length(model$cat_predictors)) {
    x_cat <- matrix(0, nrow = nrow(synth), ncol = length(model$cat_predictors))

    for(j in seq_along(model$cat_predictors)) {
      x_cat[, j] <- as.integer(synth[[model$cat_predictors[j]]])
    }
  }
  else x_cat <- matrix(integer(0), nrow = nrow(synth), ncol = 0)

  data[[model$target]][call_tree_engine("draw", list(
    model = model$model,
    x_num = x_num,
    x_cat = x_cat
  ))]
}

calibrate <- function(data, synth_l) {
  for(target in names(data)) {
    if(is.numeric(data[[target]])) {
      values <- unlist(lapply(synth_l, `[[`, target), use.names = FALSE)
      value_levels <- sort(unique(values))
      value_counts <- tabulate(match(values, value_levels), length(value_levels))
      upper <- cumsum(value_counts) / length(values)
      lower <- c(0, head(upper, -1))
      target_values <- sort(data[[target]])

      for(synth_id in seq_along(synth_l)) {
        value_ids <- match(synth_l[[synth_id]][[target]], value_levels)
        probs <- lower[value_ids] + runif(length(value_ids)) * (upper[value_ids] - lower[value_ids])
        synth_l[[synth_id]][[target]] <- target_values[pmax(1, ceiling(probs * length(target_values)))]
      }
    }
    else {
      original_counts <- tabulate(as.integer(data[[target]]), nlevels(data[[target]]))
      original_probs <- original_counts / sum(original_counts)

      synth_int <- unlist(lapply(synth_l, function(x) as.integer(x[[target]])), use.names = FALSE)
      synth_counts <- tabulate(synth_int, nlevels(data[[target]]))
      synth_probs <- synth_counts / sum(synth_counts)

      transition <- diag(nlevels(data[[target]]))
      excess <- pmax(synth_probs - original_probs, 0)
      shortfall <- pmax(original_probs - synth_probs, 0)

      for(value_id in which(excess > 0)) {
        transition[value_id, value_id] <- original_probs[value_id] / synth_probs[value_id]
        transition[value_id, which(shortfall > 0)] <- excess[value_id] * shortfall[shortfall > 0] / (synth_probs[value_id] * sum(shortfall))
      }

      for(synth_id in seq_along(synth_l)) {
        synth_int <- as.integer(synth_l[[synth_id]][[target]])
        values <- synth_int

        for(value_id in seq_len(nlevels(data[[target]]))) {
          rows <- which(synth_int == value_id)
          if(length(rows)) values[rows] <- sample.int(nlevels(data[[target]]), length(rows), replace = TRUE, prob = transition[value_id, ])
        }

        synth_l[[synth_id]][[target]] <- factor(levels(data[[target]])[values], levels = levels(data[[target]]), ordered = is.ordered(data[[target]]))
      }
    }
  }

  synth_l
}

data("prostate")
prostate$train <- NULL
prostate[] <- lapply(prostate, function(x) if(is.integer(x)) as.numeric(x) else x)
prostate$svi <- factor(prostate$svi)
prostate <- clean_names(prostate)

data("supraclavicular")
supra <- supraclavicular
rm(supraclavicular)
supra$subject <- NULL
supra$group <- factor(supra$group)
supra$gender <- factor(supra$gender)
supra$med_censor <- factor(supra$med_censor)
supra$nerve_block_censor <- NULL
supra <- clean_names(supra)
supra <- na.omit(supra)

data("blood_storage")
blood <- blood_storage
rm(blood_storage)
blood$AdjRadTherapy <- NULL
blood[] <- lapply(blood, function(x) if(length(unique(x[!is.na(x)])) <= 4) factor(x) else x)
blood <- clean_names(blood)
blood <- na.omit(blood)

wbcd <- readRDS(paste0(wd, "wbcd.rds"))
wbcd[] <- lapply(wbcd, function(x) if(is.integer(x)) as.numeric(x) else x)
wbcd <- na.omit(wbcd)

data("AlzheimerDisease")
ad <- predictors
rm(predictors)
ad$diag <- diagnosis
rm(diagnosis)
ad[] <- lapply(ad, function(x) if(is.integer(x)) as.numeric(x) else x)
ad$male <- factor(ad$male)
ad <- na.omit(ad)

hepatitis <- readRDS(paste0(wd, "hepatitis.rds"))
hepatitis <- na.omit(hepatitis)

df_l <- list(supra = supra, prostate = prostate, blood = blood, hepatitis = hepatitis, wbcd = wbcd, ad = ad)

df_out <- data.frame(
  dataset = names(df_l),
  n_obs = sapply(df_l, nrow),
  n_num = sapply(df_l, function(x) sum(sapply(x, is.numeric))),
  n_cat = sapply(df_l, function(x) sum(sapply(x, function(v) is.factor(v) || is.character(v))))
)
df_out[order(df_out$n_obs), ]

prop_grid <- seq(from = 0.02, to = 0.98, length.out = 7)
synth_count <- 100
visit_count <- 20

method_plan <- rbind(
  data.frame(method = "sequential_tree", spec = "seq", model = "tree", calibrated = FALSE, param = "minbucket", stringsAsFactors = FALSE),
  data.frame(method = "sequential_tree_calibrated", spec = "seq", model = "tree", calibrated = TRUE, param = "minbucket", stringsAsFactors = FALSE),
  data.frame(method = "sequential_subagging", spec = "seq", model = "subagging", calibrated = FALSE, param = "prop", stringsAsFactors = FALSE),
  data.frame(method = "sequential_subagging_calibrated", spec = "seq", model = "subagging", calibrated = TRUE, param = "prop", stringsAsFactors = FALSE),
  data.frame(method = "fcs_tree", spec = "fcs", model = "tree", calibrated = FALSE, param = "minbucket", stringsAsFactors = FALSE),
  data.frame(method = "fcs_subagging", spec = "fcs", model = "subagging", calibrated = FALSE, param = "prop", stringsAsFactors = FALSE),
  data.frame(method = "fcs_tree_calibrated", spec = "fcs", model = "tree", calibrated = TRUE, param = "minbucket", stringsAsFactors = FALSE),
  data.frame(method = "fcs_subagging_calibrated", spec = "fcs", model = "subagging", calibrated = TRUE, param = "prop", stringsAsFactors = FALSE)
)

result_list <- list()
result_index <- 0

run_one_method_plan <- function(plan_id, data_id, data_name, data_current, visit_orders, minbucket_grid, prop_grid, synth_count, visit_count) {
  plan_row <- method_plan[plan_id, , drop = FALSE]
  param_values <- if(plan_row$param == "minbucket") minbucket_grid else prop_grid

  out <- vector("list", length(param_values) * visit_count)
  out_id <- 0

  for(param_value in param_values) {
    minbucket_value <- if(plan_row$param == "minbucket") as.integer(param_value) else 1
    prop_value <- if(plan_row$param == "prop") as.numeric(param_value) else 1
    ntrees_value <- if(plan_row$model == "subagging") 100 else 1
    synth_by_visit <- vector("list", visit_count)

    if(plan_row$spec == "seq") {
      for(visit_id in seq_len(visit_count)) {
        visit_order <- visit_orders[[visit_id]]

        models <- vector("list", length(visit_order))
        names(models) <- visit_order

        for(position in seq_along(visit_order)) {
          target <- visit_order[position]
          models[[target]] <- fit_cond(
            data = data_current,
            target = target,
            predictors = visit_order[seq_len(position - 1)],
            minbucket = minbucket_value,
            prop = prop_value,
            ntrees = ntrees_value
          )
        }

        synth_l <- vector("list", synth_count)

        for(synth_id in seq_len(synth_count)) {
          synth <- data_current

          for(target in names(data_current)) {
            synth[[target]] <- data_current[[target]][sample.int(nrow(data_current), nrow(data_current), replace = TRUE)]
          }

          for(target in visit_order) {
            synth[[target]] <- draw_cond(models[[target]], data_current, synth)
          }

          synth_l[[synth_id]] <- synth
        }

        if(isTRUE(plan_row$calibrated)) synth_l <- calibrate(data_current, synth_l)

        synth_by_visit[[visit_id]] <- synth_l
      }
    }
    else {
      visit_sequences <- vector("list", visit_count * synth_count)

      for(visit_id in seq_len(visit_count)) {
        for(synth_id in seq_len(synth_count)) {
          visit_sequences[[(visit_id - 1) * synth_count + synth_id]] <- visit_orders[[visit_id]]
        }
      }

      models <- vector("list", ncol(data_current))
      names(models) <- names(data_current)

      for(target in names(data_current)) {
        models[[target]] <- fit_cond(
          data = data_current,
          target = target,
          predictors = setdiff(names(data_current), target),
          minbucket = minbucket_value,
          prop = prop_value,
          ntrees = ntrees_value
        )
      }

      synth_all <- vector("list", length(visit_sequences))

      for(synth_id in seq_along(visit_sequences)) {
        synth <- as.data.frame(lapply(data_current, function(x) sample(x, size = length(x), replace = TRUE)))

        for(target in names(data_current)) {
          synth[[target]] <- data_current[[target]][sample.int(nrow(data_current), nrow(data_current), replace = TRUE)]
        }

        for(iteration in seq_len(10)) {
          for(target in visit_sequences[[synth_id]]) {
            synth[[target]] <- draw_cond(models[[target]], data_current, synth)
          }
        }

        synth_all[[synth_id]] <- synth
      }

      for(visit_id in seq_len(visit_count)) {
        synth_ids <- ((visit_id - 1) * synth_count + 1):(visit_id * synth_count)
        synth_block <- synth_all[synth_ids]

        if(isTRUE(plan_row$calibrated)) synth_block <- calibrate(data_current, synth_block)

        synth_by_visit[[visit_id]] <- synth_block
      }
    }

    for(visit_id in seq_len(visit_count)) {
      match_values <- numeric(length(synth_by_visit[[visit_id]]))
      score_values <- numeric(length(synth_by_visit[[visit_id]]))
      score_values_marginal <- numeric(length(synth_by_visit[[visit_id]]))

      for(synth_id in seq_along(synth_by_visit[[visit_id]])) {
        match_values[synth_id] <- adaptive_matches_prop(data_current, synth_by_visit[[visit_id]][[synth_id]])
        score_values[synth_id] <- 1 - tsAUC(data_current, synth_by_visit[[visit_id]][[synth_id]])
        score_values_marginal[synth_id] <- -marginal_T(data_current, synth_by_visit[[visit_id]][[synth_id]])
      }

      out_id <- out_id + 1

      out[[out_id]] <- data.frame(
        data_id = data_id,
        data_name = data_name,
        method = plan_row$method,
        spec = plan_row$spec,
        model = plan_row$model,
        calibrated = plan_row$calibrated,
        param = plan_row$param,
        value = param_value,
        visit = visit_id,
        visit.sequence = paste(visit_orders[[visit_id]], collapse = "-"),
        priv = mean(match_values),
        util = mean(score_values),
        util_marg = mean(score_values_marginal),
        stringsAsFactors = FALSE
      )
    }
  }

  do.call(rbind, out[seq_len(out_id)])
}

workers <- min(nrow(method_plan), parallel::detectCores() - 1)
cl <- parallel::makeCluster(workers)
parallel::clusterExport(
  cl,
  c("tree_engine_includes", "tree_engine_code", "compile_tree_engine"),
  envir = .GlobalEnv
)
parallel::clusterEvalQ(cl, {
  library(Rcpp)
  compile_tree_engine()
  NULL
})
future::plan(future::cluster, workers = cl)

set.seed(1)

for(data_id in seq_along(df_l)) {
  print(names(df_l)[data_id])

  data_current <- df_l[[data_id]]

  minbucket_grid <- round(seq(from = 0.02, to = 0.5, length.out = 7) * nrow(data_current))

  visit_orders <- vector("list", visit_count)

  for(visit_id in seq_len(visit_count)) {
    visit_orders[[visit_id]] <- names(data_current)[sample.int(ncol(data_current))]
  }

  plan_results <- future_lapply(
    seq_len(nrow(method_plan)),
    run_one_method_plan,
    data_id = data_id,
    data_name = names(df_l)[data_id],
    data_current = data_current,
    visit_orders = visit_orders,
    minbucket_grid = minbucket_grid,
    prop_grid = prop_grid,
    synth_count = synth_count,
    visit_count = visit_count,
    future.seed = TRUE
  )

  result_list <- c(result_list, plan_results)
}

method_labels <- c(
  sequential_tree = "Sequential (trees)",
  sequential_tree_calibrated = "Sequential (trees)",
  sequential_subagging = "Sequential (subagging)",
  sequential_subagging_calibrated = "Sequential (subagging)",
  fcs_tree = "FCS (trees)",
  fcs_tree_calibrated = "FCS (trees)",
  fcs_subagging = "FCS (subagging)",
  fcs_subagging_calibrated = "FCS (subagging)"
)
method_colours_common <- c(
  sequential_tree = bor_cols()[1],
  sequential_tree_calibrated = bor_cols()[1],
  sequential_subagging = bor_cols()[2],
  sequential_subagging_calibrated = bor_cols()[2],
  fcs_tree = bor_cols()[3],
  fcs_tree_calibrated = bor_cols()[3],
  fcs_subagging = bor_cols()[4],
  fcs_subagging_calibrated = bor_cols()[4]
)

plot_results <- function(plot_data, metric, calibrated = T) {
  plot_data <- plot_data[plot_data$calibrated == calibrated, , drop = FALSE]
  plot_data$data_name <- factor(plot_data$data_name, levels = df_out$dataset[order(df_out$n_obs)])

  plot_data$util_plot <- plot_data[[metric]]
  plot_data <- plot_data[order(plot_data$data_name, plot_data$method, plot_data$visit, plot_data$value), , drop = FALSE]

  mean_data <- aggregate(
    cbind(priv, util_plot) ~ data_id + data_name + method + spec + model + calibrated + param + value,
    data = plot_data,
    FUN = mean
  )

  mean_data <- mean_data[order(mean_data$data_name, mean_data$method, mean_data$value), , drop = FALSE]
  xlab <- if(metric == "util") "ts-AUC" else "marginal T"

  method_breaks <- intersect(names(method_labels), unique(plot_data$method))
  plot_data_seq <- plot_data[plot_data$spec == "seq", , drop = FALSE]
  plot_data_fcs <- plot_data[plot_data$spec == "fcs", , drop = FALSE]

  mean_data_seq <- mean_data[mean_data$spec == "seq", , drop = FALSE]
  mean_data_fcs <- mean_data[mean_data$spec == "fcs", , drop = FALSE]

  ggplot() +
    geom_path(data = plot_data_seq, aes(x = util_plot, y = priv, group = interaction(data_id, method, visit), colour = method), alpha = .25, linewidth = .3) +
    geom_path(data = plot_data_fcs, aes(x = util_plot, y = priv, group = interaction(data_id, method, visit), colour = method), alpha = .25, linewidth = .3) +
    geom_path(data = mean_data_seq, aes(x = util_plot, y = priv, group = interaction(data_id, method), colour = method), linewidth = 1.2) +
    geom_path(data = mean_data_fcs, aes(x = util_plot, y = priv, group = interaction(data_id, method), colour = method), linewidth = 1.2) +
    facet_wrap(~ data_name, ncol = 3, scales = "free", labeller = labeller(data_name = as_labeller(c(
      supra = "Supraclavicular",
      prostate = "Prostate",
      blood = "Blood storage",
      hepatitis = "Hepatitis",
      wbcd = "WBCD",
      ad = "Alzheimer's disease"
    )))) +
    labs(x = xlab, y = "aPMR", colour = "method") +
    scale_colour_manual(
      values = method_colours_common,
      breaks = method_breaks,
      labels = method_labels[method_breaks]
    ) +
    theme_minimal() +
    theme(strip.text = element_text(size = 13), legend.position = "bottom")
}

results <- do.call(rbind, result_list)
plot_results(results, "util", F)
save_plot(save_dir, 10.5, 5.5)
plot_results(results, "util", T)
save_plot(save_dir, 10.5, 5.5)
plot_results(results, "util_marg", F)
save_plot(save_dir, 10.5, 5.5)
plot_results(results, "util_marg", T)
save_plot(save_dir, 10.5, 5.5)

plot_visit_sd <- function(plot_data, metric, ylabel = "") {
  plot_data$metric_sd <- plot_data[[metric]]

  sd_data <- aggregate(
    metric_sd ~ data_name + method + spec + calibrated + model + value,
    data = plot_data,
    FUN = sd
  )

  sd_data <- aggregate(
    metric_sd ~ data_name + spec + calibrated + model,
    data = sd_data,
    FUN = mean
  )

  sd_data$spec <- ifelse(
    sd_data$spec == "seq",
    "Sequential",
    "FCS"
  )

  sd_data$model <- factor(
    sd_data$model,
    levels = c("tree", "subagging"),
    labels = c("trees", "subagging")
  )

  sd_data$spec <- factor(
    paste0(sd_data$spec, " (", sd_data$model, ")"),
    levels = c(
      "Sequential (trees)",
      "Sequential (subagging)",
      "FCS (trees)",
      "FCS (subagging)"
    )
  )

  sd_data$calibrated <- factor(
    ifelse(sd_data$calibrated, "Yes", "No"),
    levels = c("No", "Yes")
  )

  sd_data$data_name <- factor(
    sd_data$data_name,
    levels = df_out$dataset[order(df_out$n_obs)]
  )

  ggplot(sd_data, aes(x = data_name, y = metric_sd, fill = spec, pattern = calibrated)) +
    geom_col_pattern(
      position = position_dodge(width = .8),
      width = .7,
      colour = "black",
      linewidth = .25,
      pattern_fill = "white",
      pattern_colour = "white",
      pattern_density = .1,
      pattern_spacing = .03
    ) +
    scale_x_discrete(
      labels = c(
        supra = "Supraclavicular",
        prostate = "Prostate",
        blood = "Blood storage",
        hepatitis = "Hepatitis",
        wbcd = "WBCD",
        ad = "Alzheimer's\ndisease"
      )[df_out$dataset[order(df_out$n_obs)]]
    ) +
    scale_fill_manual(
      name = "Configuration",
      values = c(
        "Sequential (trees)" = bor_cols()[1],
        "Sequential (subagging)" = bor_cols()[2],
        "FCS (trees)" = bor_cols()[3],
        "FCS (subagging)" = bor_cols()[4]
      )
    ) +
    scale_pattern_manual(
      name = "Marginal calibration",
      values = c(
        "No" = "circle",
        "Yes" = "stripe"
      )
    ) +
    guides(
      fill = guide_legend(
        title.position = "top",
        override.aes = list(pattern = "none"),
        nrow = 2,
        byrow = TRUE,
        order = 1
      ),
      pattern = guide_legend(
        title.position = "top",
        order = 2
      )
    ) +
    labs(
      x = NULL,
      y = ylabel
    ) +
    theme_minimal() +
    theme(
      legend.title = element_text(hjust = .5),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor.x = ggplot2::element_blank()
    )
}

p_util <- plot_visit_sd(results, "util", "ts-AUC (sd)")

p_priv <- plot_visit_sd(results, "priv", "aPMR (sd)") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

(p_priv / p_util) + plot_layout(guides = "collect") & theme(legend.position = "bottom")

save_plot(save_dir, 10, 5.5)


###################################################


set.seed(2)
opt_util <- sapply(df_l, function(df) 0.5)

worst_util <- sapply(df_l, function(df) {
  mean(sapply(naive_synthesis(df, p = 0, m = synth_count), function(s) {
    1 - tsAUC(df, s)
  }))
})

opt_priv <- sapply(df_l, function(df) {
  mean(sapply(naive_synthesis(df, p = 0, m = synth_count), function(s) {
    adaptive_matches_prop(df, s)
  }))
})

worst_priv <- sapply(df_l, function(df) 1)

make_table <- function() {
  datasets <- names(df_l)[order(df_out$n_obs)]

  dataset_labels <- c(prostate = "Pr", supra = "Sc", blood = "Bs", ad = "Ad", hepatitis = "He", wbcd = "WB")

  methods <- c(
    "sequential_tree",
    "sequential_tree_calibrated",
    "sequential_subagging",
    "sequential_subagging_calibrated",
    "fcs_tree",
    "fcs_tree_calibrated",
    "fcs_subagging",
    "fcs_subagging_calibrated"
  )

  standardize <- function(x, opt, worst, dataset) {
    1 - (unname(opt[dataset]) - x) / (unname(opt[dataset]) - unname(worst[dataset]))
  }

  score_means <- matrix(NA_real_, nrow = length(methods), ncol = length(datasets), dimnames = list(methods, datasets))
  score_sds <- matrix(NA_real_, nrow = length(methods), ncol = length(datasets), dimnames = list(methods, datasets))

  for(method in methods) {
    for(dataset in datasets) {
      sub <- results[results$method == method & results$data_name == dataset, , drop = FALSE]

      sub$priv_std <- standardize(sub$priv, opt_priv, worst_priv, dataset)
      sub$util_std <- standardize(sub[["util"]], opt_util, worst_util, dataset)
      sub$score <- (sub$priv_std + sub$util_std) / 2

      score_by_value <- aggregate(score ~ value, data = sub, FUN = mean)
      best_value <- score_by_value$value[which.max(score_by_value$score)]
      scores <- sub$score[sub$value == best_value]

      score_means[method, dataset] <- mean(scores)
      score_sds[method, dataset] <- sd(scores)
    }
  }

  br <- "\\\\"

  cell <- function(mean_value, sd_value, highlight) {
    if(highlight) {
      paste0(
        "\\SetCell{bg=yellow!40}\\makecell{\\textbf{",
        sprintf("%.3f", mean_value),
        "}",
        br,
        "\\textbf{(",
        sprintf("%.3f", sd_value),
        ")}}"
      )
    }
    else {
      paste0(
        "\\makecell{",
        sprintf("%.3f", mean_value),
        br,
        "(",
        sprintf("%.3f", sd_value),
        ")}"
      )
    }
  }

  equivalent_methods <- setNames(
    ifelse(
      startsWith(methods, "sequential_"),
      sub("^sequential_", "fcs_", methods),
      sub("^fcs_", "sequential_", methods)
    ),
    methods
  )

  values <- matrix("", nrow = length(methods), ncol = length(datasets), dimnames = list(methods, datasets))

  for(dataset in datasets) {
    for(method in methods) {
      highlight <- score_means[method, dataset] > score_means[equivalent_methods[[method]], dataset]

      values[method, dataset] <- cell(
        score_means[method, dataset],
        score_sds[method, dataset],
        highlight
      )
    }
  }

  method_labels <- c(
    sequential_tree = "\\textbf{Trees}",
    sequential_tree_calibrated = paste0("\\makecell{\\textbf{Trees}", br, "\\textbf{(calibrated)}}"),
    sequential_subagging = "\\textbf{Subagging}",
    sequential_subagging_calibrated = paste0("\\makecell{\\textbf{Subagging}", br, "\\textbf{(calibrated)}}"),
    fcs_tree = "\\textbf{Trees}",
    fcs_tree_calibrated = paste0("\\makecell{\\textbf{Trees}", br, "\\textbf{(calibrated)}}"),
    fcs_subagging = "\\textbf{Subagging}",
    fcs_subagging_calibrated = paste0("\\makecell{\\textbf{Subagging}", br, "\\textbf{(calibrated)}}")
  )

  row <- function(method, group = NULL) {
    first_cell <- if(is.null(group)) {
      "    & "
    }
    else {
      paste0(
        "    \\SetCell[r=",
        4,
        "]{c,m}\\rotatebox[origin=c]{90}{\\textbf{",
        group,
        "}} & "
      )
    }

    paste(
      paste0(first_cell, method_labels[method]),
      paste0(
        "    & ",
        values[method, datasets],
        c(rep("", length(datasets) - 1), " \\\\"),
        collapse = "\n"
      ),
      sep = "\n"
    )
  }

  paste(
    paste0("    \\SetCell[c=2]{c}\\textbf{Method} & & ", paste0("\\textbf{", dataset_labels[datasets], "}", collapse = " & "), " \\\\"),
    row("sequential_tree", "Sequential"),
    row("sequential_tree_calibrated"),
    row("sequential_subagging"),
    row("sequential_subagging_calibrated"),
    row("fcs_tree", "FCS"),
    row("fcs_tree_calibrated"),
    row("fcs_subagging"),
    row("fcs_subagging_calibrated"),
    sep = "\n"
  )
}

cat(make_table())


###################################################


nscan <- 10
scan_result_list <- list()

set.seed(3)

visit_orders_by_data <- vector("list", length(df_l))
names(visit_orders_by_data) <- names(df_l)

for(data_id in seq_along(df_l)) {
  data_current <- df_l[[data_id]]
  visit_orders <- vector("list", visit_count)

  for(visit_id in seq_len(visit_count)) {
    visit_orders[[visit_id]] <- names(data_current)[sample.int(ncol(data_current))]
  }

  visit_orders_by_data[[data_id]] <- visit_orders
}

scan_jobs <- do.call(
  rbind,
  lapply(seq_along(df_l), function(data_id) {
    data.frame(
      data_id = data_id,
      model_name = c("tree", "subagging"),
      stringsAsFactors = FALSE
    )
  })
)

run_one_scan_job <- function(job_id, scan_jobs, df_l, visit_orders_by_data, nscan, synth_count, visit_count) {
  if(!exists("tree_engine", envir = .GlobalEnv, inherits = FALSE)) {
    compile_tree_engine()
  }

  data_id <- scan_jobs$data_id[job_id]
  model_name <- scan_jobs$model_name[job_id]
  data_current <- df_l[[data_id]]
  data_name <- names(df_l)[data_id]
  visit_orders <- visit_orders_by_data[[data_id]]

  models <- vector("list", ncol(data_current))
  names(models) <- names(data_current)

  for(target in names(data_current)) {
    models[[target]] <- fit_cond(
      data = data_current,
      target = target,
      predictors = setdiff(names(data_current), target),
      minbucket = if(model_name == "tree") round(0.25 * nrow(data_current)) else 1,
      prop = if(model_name == "subagging") 0.5 else 1,
      ntrees = if(model_name == "subagging") 100 else 1
    )
  }

  out <- vector("list", visit_count * synth_count * (nscan + 1))
  out_id <- 0

  for(visit_id in seq_len(visit_count)) {
    for(synth_id in seq_len(synth_count)) {
      synth <- data_current

      for(target in names(data_current)) {
        synth[[target]] <- data_current[[target]][sample.int(nrow(data_current), nrow(data_current), replace = TRUE)]
      }

      for(scan in 0:nscan) {
        if(scan > 0) {
          for(target in visit_orders[[visit_id]]) {
            synth[[target]] <- draw_cond(models[[target]], data_current, synth)
          }
        }

        priv <- adaptive_matches_prop(data_current, synth)
        util <- 1 - tsAUC(data_current, synth)
        util_marg <- -marginal_T(data_current, synth)

        if(scan == 0) {
          baseline_priv <- priv
          baseline_util <- util
          baseline_util_marg <- util_marg
        }

        out_id <- out_id + 1

        out[[out_id]] <- data.frame(
          data_id = data_id,
          data_name = data_name,
          model = model_name,
          visit = visit_id,
          synth = synth_id,
          scan = scan,
          priv = priv - baseline_priv,
          util = util - baseline_util,
          util_marg = util_marg - baseline_util_marg,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  do.call(rbind, out[seq_len(out_id)])
}

future::plan(future::sequential)
parallel::stopCluster(cl)
workers <- parallel::detectCores() - 1
cl <- parallel::makeCluster(workers)
parallel::clusterExport(
  cl,
  c("tree_engine_includes", "tree_engine_code", "compile_tree_engine"),
  envir = .GlobalEnv
)
parallel::clusterEvalQ(cl, {
  library(Rcpp)
  compile_tree_engine()
  NULL
})
future::plan("cluster", workers = cl)

scan_result_list <- future_lapply(
  seq_len(nrow(scan_jobs)),
  run_one_scan_job,
  scan_jobs = scan_jobs,
  df_l = df_l,
  visit_orders_by_data = visit_orders_by_data,
  nscan = nscan,
  synth_count = synth_count,
  visit_count = visit_count,
  future.seed = TRUE
)

future::plan(future::sequential)
parallel::stopCluster(cl)

scan_results <- do.call(rbind, scan_result_list)

plot_scan_results <- function(data, metric_name, model_name) {
  data <- data[data$model == model_name, c("data_id", "data_name", "scan", metric_name), drop = FALSE]
  names(data)[4] <- "value"

  data <- aggregate(
    value ~ data_id + data_name + scan,
    data = data,
    FUN = mean
  )

  names(data)[names(data) == "value"] <- "mean"

  for(i in split(seq_len(nrow(data)), data$data_id)) {
    min_i <- min(data$mean[i], na.rm = TRUE)
    max_i <- max(data$mean[i], na.rm = TRUE)
    range_i <- max_i - min_i

    if(is.finite(range_i) && range_i > 0) data$mean[i] <- (data$mean[i] - min_i) / range_i
    else data$mean[i] <- 0
  }

  data$data_name <- factor(data$data_name, levels = df_out$dataset[order(df_out$n_obs)])

  ggplot(data, aes(x = scan, y = mean, group = data_name, colour = data_name, fill = data_name, shape = data_name)) +
    lapply(rev(levels(data$data_name)), function(x) {
      geom_path(data = data[data$data_name == x, , drop = FALSE], linewidth = 0.8)
    }) +
    lapply(rev(levels(data$data_name)), function(x) {
      geom_point(data = data[data$data_name == x, , drop = FALSE], size = 2.5)
    }) +
    scale_x_continuous(breaks = seq(0, nscan, by = 2)) +
    scale_colour_manual(
      limits = df_out$dataset[order(df_out$n_obs)],
      labels = c(
        supra = "Supraclavicular",
        prostate = "Prostate",
        blood = "Blood storage",
        hepatitis = "Hepatitis",
        wbcd = "WBCD",
        ad = "Alzheimer's disease"
      )[df_out$dataset[order(df_out$n_obs)]],
      values = bor_cols()
    ) +
    scale_fill_manual(
      limits = df_out$dataset[order(df_out$n_obs)],
      labels = c(
        supra = "Supraclavicular",
        prostate = "Prostate",
        blood = "Blood storage",
        hepatitis = "Hepatitis",
        wbcd = "WBCD",
        ad = "Alzheimer's disease"
      )[df_out$dataset[order(df_out$n_obs)]],
      values = bor_cols()
    ) +
    scale_shape_manual(
      limits = df_out$dataset[order(df_out$n_obs)],
      labels = c(
        supra = "Supraclavicular",
        prostate = "Prostate",
        blood = "Blood storage",
        hepatitis = "Hepatitis",
        wbcd = "WBCD",
        ad = "Alzheimer's disease"
      )[df_out$dataset[order(df_out$n_obs)]],
      values = c(
        supra = 16,
        prostate = 17,
        blood = 15,
        hepatitis = 18,
        wbcd = 25,
        ad = 4
      )[df_out$dataset[order(df_out$n_obs)]]
    ) +
    labs(
      x = "Iteration",
      y = switch(
        metric_name,
        priv = "aPMR",
        util = "ts-AUC",
        util_marg = "marginal T"
      ),
      colour = "dataset",
      shape = "dataset",
      fill = "dataset"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

wrap_plots(
  plot_scan_results(scan_results, "priv", "tree") + labs(title = "Individual trees"),
  plot_scan_results(scan_results, "priv", "subagging") + labs(title = "Subagging"),
  plot_scan_results(scan_results, "util", "tree"),
  plot_scan_results(scan_results, "util", "subagging"),
  plot_scan_results(scan_results, "util_marg", "tree"),
  plot_scan_results(scan_results, "util_marg", "subagging"),
  ncol = 2,
  byrow = TRUE,
  guides = "collect",
  axis_titles = "collect"
) &
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, size = 16, margin = ggplot2::margin(b = 12))
  )


save_plot(save_dir, 9, 6)


###################################################


run_one_c2st_data <- function(data_id, df_l, prop_grid, synth_count, visit_count) {
  c2st_rf <- function(synth_l) {
    data <- do.call(rbind, synth_l)
    source <- factor(rep(seq_along(synth_l), vapply(synth_l, nrow, integer(1))))

    model <- randomForest(
      x = data,
      y = source,
      ntree = 100
    )

    1 - tail(model$err.rate[, "OOB"], 1)
  }

  data_current <- df_l[[data_id]]
  visit_orders <- vector("list", visit_count)

  for(visit_id in seq_len(visit_count)) {
    visit_orders[[visit_id]] <- names(data_current)[sample.int(ncol(data_current))]
  }

  out <- vector("list", length(prop_grid))

  for(prop_id in seq_along(prop_grid)) {
    models <- vector("list", ncol(data_current))
    names(models) <- names(data_current)

    for(target in names(data_current)) {
      models[[target]] <- fit_cond(
        data = data_current,
        target = target,
        predictors = setdiff(names(data_current), target),
        minbucket = 1,
        prop = prop_grid[prop_id],
        ntrees = 100
      )
    }

    synth_by_visit <- vector("list", visit_count)

    for(visit_id in seq_len(visit_count)) {
      synth_l <- vector("list", synth_count)

      for(synth_id in seq_len(synth_count)) {
        synth <- data_current

        for(target in names(data_current)) {
          synth[[target]] <- data_current[[target]][sample.int(nrow(data_current), nrow(data_current), replace = TRUE)]
        }

        for(iteration in seq_len(10)) {
          for(target in visit_orders[[visit_id]]) {
            synth[[target]] <- draw_cond(models[[target]], data_current, synth)
          }
        }

        synth_l[[synth_id]] <- synth
      }

      synth_by_visit[[visit_id]] <- do.call(rbind, calibrate(data_current, synth_l))
    }

    out[[prop_id]] <- data.frame(
      data_name = names(df_l)[data_id],
      prop = prop_grid[prop_id],
      statistic = c2st_rf(synth_by_visit),
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, out)
}

set.seed(4)

workers <- min(length(df_l), parallel::detectCores() - 1)
cl <- parallel::makeCluster(workers)

parallel::clusterExport(
  cl,
  c("tree_engine_includes", "tree_engine_code", "compile_tree_engine"),
  envir = .GlobalEnv
)

parallel::clusterEvalQ(cl, {
  library(Rcpp)
  compile_tree_engine()
  NULL
})

future::plan(future::cluster, workers = cl)

c2st_results <- do.call(
  rbind,
  future.apply::future_lapply(
    seq_along(df_l),
    run_one_c2st_data,
    df_l = df_l,
    prop_grid = prop_grid,
    synth_count = synth_count,
    visit_count = visit_count,
    future.seed = TRUE
  )
)

future::plan(future::sequential)
parallel::stopCluster(cl)

c2st_results$data_name <- factor(
  c2st_results$data_name,
  levels = df_out$dataset[order(df_out$n_obs)]
)

ggplot(c2st_results, aes(x = prop, y = statistic, colour = data_name, shape = data_name, fill = data_name)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = prop_grid) +
  scale_colour_manual(
    limits = df_out$dataset[order(df_out$n_obs)],
    labels = c(
      supra = "Supraclavicular",
      prostate = "Prostate",
      blood = "Blood storage",
      hepatitis = "Hepatitis",
      wbcd = "WBCD",
      ad = "Alzheimer's disease"
    )[df_out$dataset[order(df_out$n_obs)]],
    values = bor_cols()
  ) +
  scale_fill_manual(
    limits = df_out$dataset[order(df_out$n_obs)],
    labels = c(
      supra = "Supraclavicular",
      prostate = "Prostate",
      blood = "Blood storage",
      hepatitis = "Hepatitis",
      wbcd = "WBCD",
      ad = "Alzheimer's disease"
    )[df_out$dataset[order(df_out$n_obs)]],
    values = bor_cols()
  ) +
  scale_shape_manual(
    limits = df_out$dataset[order(df_out$n_obs)],
    labels = c(
      supra = "Supraclavicular",
      prostate = "Prostate",
      blood = "Blood storage",
      hepatitis = "Hepatitis",
      wbcd = "WBCD",
      ad = "Alzheimer's disease"
    )[df_out$dataset[order(df_out$n_obs)]],
    values = c(
      supra = 16,
      prostate = 17,
      blood = 15,
      hepatitis = 18,
      wbcd = 25,
      ad = 4
    )[df_out$dataset[order(df_out$n_obs)]]
  ) +
  labs(
    x = expression(pi),
    y = "OOB accuracy",
    colour = "Dataset",
    shape = "Dataset",
    fill = "Dataset"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

save_plot(save_dir, 6, 3.5)


###################################################


run_one_bootstrap_data <- function(data_id, df_l, visit_orders_by_data, prop_grid, synth_count, visit_count) {
  if(!exists("tree_engine", envir = .GlobalEnv, inherits = FALSE)) {
    compile_tree_engine()
  }

  data_current <- df_l[[data_id]]
  visit_orders <- visit_orders_by_data[[data_id]]
  out <- vector("list", length(prop_grid) * visit_count)
  out_id <- 0

  for(prop_value in prop_grid) {
    models <- vector("list", ncol(data_current))
    names(models) <- names(data_current)

    for(target in names(data_current)) {
      models[[target]] <- fit_cond(
        data = data_current,
        target = target,
        predictors = setdiff(names(data_current), target),
        minbucket = 1,
        prop = prop_value,
        ntrees = 100
      )
    }

    for(visit_id in seq_len(visit_count)) {
      synth_l <- vector("list", synth_count)

      for(synth_id in seq_len(synth_count)) {
        synth <- data_current[sample.int(nrow(data_current), nrow(data_current), replace = TRUE), , drop = FALSE]

        for(iteration in seq_len(10)) {
          for(target in visit_orders[[visit_id]]) {
            synth[[target]] <- draw_cond(models[[target]], data_current, synth)
          }
        }

        synth_l[[synth_id]] <- synth
      }

      synth_l <- calibrate(data_current, synth_l)
      match_values <- numeric(synth_count)
      score_values <- numeric(synth_count)

      for(synth_id in seq_along(synth_l)) {
        match_values[synth_id] <- adaptive_matches_prop(data_current, synth_l[[synth_id]])
        score_values[synth_id] <- 1 - tsAUC(data_current, synth_l[[synth_id]])
      }

      out_id <- out_id + 1

      out[[out_id]] <- data.frame(
        data_name = names(df_l)[data_id],
        value = prop_value,
        visit = visit_id,
        priv = mean(match_values),
        util = mean(score_values),
        initialization = "Bootstrap",
        stringsAsFactors = FALSE
      )
    }
  }

  do.call(rbind, out)
}

visit_orders_by_data <- lapply(seq_along(df_l), function(data_id) {
  visit_data <- unique(results[
    results$data_id == data_id & results$method == "fcs_subagging_calibrated",
    c("visit", "visit.sequence"),
    drop = FALSE
  ])

  visit_data <- visit_data[order(visit_data$visit), , drop = FALSE]
  strsplit(visit_data$visit.sequence, "-", fixed = TRUE)
})
names(visit_orders_by_data) <- names(df_l)

set.seed(5)

workers <- min(length(df_l), parallel::detectCores() - 1)
cl <- parallel::makeCluster(workers)

parallel::clusterExport(
  cl,
  c("tree_engine_includes", "tree_engine_code", "compile_tree_engine"),
  envir = .GlobalEnv
)

parallel::clusterEvalQ(cl, {
  library(Rcpp)
  compile_tree_engine()
  NULL
})

future::plan(future::cluster, workers = cl)

bootstrap_results <- do.call(
  rbind,
  future_lapply(
    seq_along(df_l),
    run_one_bootstrap_data,
    df_l = df_l,
    visit_orders_by_data = visit_orders_by_data,
    prop_grid = prop_grid,
    synth_count = synth_count,
    visit_count = visit_count,
    future.seed = TRUE
  )
)

future::plan(future::sequential)
parallel::stopCluster(cl)

marginal_results <- results[
  results$method == "fcs_subagging_calibrated",
  c("data_name", "value", "visit", "priv", "util"),
  drop = FALSE
]

marginal_results$initialization <- "Marginals"
initialization_results <- rbind(marginal_results, bootstrap_results)

mean_data <- aggregate(
  cbind(priv, util) ~ data_name + value + initialization,
  data = initialization_results,
  FUN = mean
)

mean_data$data_name <- factor(
  mean_data$data_name,
  levels = df_out$dataset[order(df_out$n_obs)]
)

mean_data$initialization <- factor(
  mean_data$initialization,
  levels = c("Marginals", "Bootstrap")
)

mean_data$value <- factor(mean_data$value, levels = prop_grid)

plot_initialization_results <- function(plot_data, metric, ylabel = "") {
  plot_data$metric_mean <- plot_data[[metric]]
  position <- position_dodge(width = .9)

  ggplot(
    plot_data,
    aes(
      x = value,
      y = metric_mean,
      group = interaction(initialization, data_name),
      fill = data_name,
      pattern = initialization
    )
  ) +
    geom_col_pattern(
      position = position,
      width = .8,
      colour = "black",
      linewidth = .25,
      pattern_fill = "white",
      pattern_colour = "white",
      pattern_density = .1,
      pattern_spacing = .03
    ) +
    scale_x_discrete(labels = format(prop_grid, nsmall = 2, trim = TRUE)) +
    scale_fill_manual(
      name = "Dataset",
      limits = df_out$dataset[order(df_out$n_obs)],
      labels = c(
        supra = "Supraclavicular",
        prostate = "Prostate",
        blood = "Blood storage",
        hepatitis = "Hepatitis",
        wbcd = "WBCD",
        ad = "Alzheimer's disease"
      )[df_out$dataset[order(df_out$n_obs)]],
      values = bor_cols()
    ) +
    scale_pattern_manual(
      name = "Initialization",
      values = c(
        Marginals = "circle",
        Bootstrap = "stripe"
      )
    ) +
    guides(
      fill = guide_legend(
        title.position = "top",
        override.aes = list(pattern = "none"),
        order = 1
      ),
      pattern = guide_legend(
        title.position = "top",
        order = 2
      )
    ) +
    labs(x = expression(pi), y = ylabel) +
    theme_minimal() +
    theme(
      legend.title = element_text(hjust = .5),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank()
    )
}

p_util <- plot_initialization_results(mean_data, "util", "ts-AUC")

p_priv <- plot_initialization_results(mean_data, "priv", "aPMR") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

(p_priv / p_util) + plot_layout(guides = "collect") & theme(legend.position = "bottom")

save_plot(save_dir, 10, 5)
