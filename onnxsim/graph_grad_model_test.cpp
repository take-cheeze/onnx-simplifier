/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Exercises graph_grad_model.{h,cpp}: the gradient of a whole model.
 *
 * Like graph_grad_test.cpp this checks structure, not numbers: nothing in this
 * build evaluates an ONNX graph. The whole-model entry point adds only
 * plumbing around BuildBackward, so the check that matters is that its
 * gradient nodes are exactly BuildBackward's nodes for the same slice.
 *
 * Plain asserts and a failure counter, like graph_grad_test.cpp.
 */
#include "graph_grad_model.h"

#include <onnx/defs/parser.h>
#include <onnx/onnx_pb.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph_grad.h"
#include "qat_graph_builder.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

// A model in ONNX text format, opset 18. Shapes are static: X[2,3] W[3,4],
// H and T[2,4], S a scalar. U is an input the loss does not use.
std::string ForwardText(const std::string& elem = "float") {
  return "<ir_version: 8, opset_import: [\"\" : 18]> "
         "g (" +
         elem + "[2, 3] X, " + elem + "[3, 4] W, " + elem + "[5] U) => (" +
         elem +
         "[] S) {\n"
         "  H = MatMul(X, W)\n"
         "  T = Tanh(H)\n"
         "  Y = Sin(U)\n"
         "  S = ReduceSum<keepdims: int = 0>(T)\n"
         "}\n";
}

onnx::ModelProto Parse(const std::string& text) {
  onnx::ModelProto model;
  onnx::OnnxParser parser(text);
  const auto status = parser.Parse(model);
  if (!status.IsOK()) {
    throw std::runtime_error("the test model does not parse: " +
                             status.ErrorMessage());
  }
  return model;
}

// Replaces the first occurrence of `from` in `text`.
std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const size_t p = text.find(from);
  if (p != std::string::npos) text.replace(p, from.size(), to);
  return text;
}

// The node that produces `name`, or nullptr.
const onnx::NodeProto* Producer(const onnx::ModelProto& m,
                                const std::string& name) {
  for (const onnx::NodeProto& nd : m.graph().node()) {
    for (const std::string& o : nd.output()) {
      if (o == name) return &nd;
    }
  }
  return nullptr;
}

const onnx::TensorProto* Initializer(const onnx::ModelProto& m,
                                     const std::string& name) {
  for (const onnx::TensorProto& t : m.graph().initializer()) {
    if (t.name() == name) return &t;
  }
  return nullptr;
}

// The values of a constant, whether it holds typed data or raw bytes.
// raw_data is little-endian whatever the host is (ONNX's rule), so decode it
// byte by byte rather than memcpy'ing host-order words (s390x is big-endian).
std::vector<int64_t> Int64Values(const onnx::TensorProto& t) {
  if (t.int64_data_size() > 0)
    return {t.int64_data().begin(), t.int64_data().end()};
  std::vector<int64_t> values;
  const std::string& raw = t.raw_data();
  for (size_t i = 0; i + 8 <= raw.size(); i += 8) {
    uint64_t bits = 0;
    for (int k = 7; k >= 0; --k) {
      bits = (bits << 8) | static_cast<unsigned char>(raw[i + k]);
    }
    values.push_back(static_cast<int64_t>(bits));
  }
  return values;
}

std::vector<float> FloatValues(const onnx::TensorProto& t) {
  if (t.float_data_size() > 0)
    return {t.float_data().begin(), t.float_data().end()};
  std::vector<float> values;
  const std::string& raw = t.raw_data();
  for (size_t i = 0; i + 4 <= raw.size(); i += 4) {
    uint32_t bits = 0;
    for (int k = 3; k >= 0; --k) {
      bits = (bits << 8) | static_cast<unsigned char>(raw[i + k]);
    }
    float v;
    std::memcpy(&v, &bits, sizeof v);
    values.push_back(v);
  }
  return values;
}

const onnx::ValueInfoProto* Output(const onnx::ModelProto& m,
                                   const std::string& name) {
  for (const onnx::ValueInfoProto& o : m.graph().output()) {
    if (o.name() == name) return &o;
  }
  return nullptr;
}

std::vector<int64_t> Dims(const onnx::ValueInfoProto& vi) {
  std::vector<int64_t> dims;
  for (const auto& d : vi.type().tensor_type().shape().dim())
    dims.push_back(d.dim_value());
  return dims;
}

// The number of forward nodes the loss needs: MatMul, the activation,
// ReduceSum.
constexpr int kForwardNodes = 3;

// The same model with Relu for Tanh. Relu's rule is plain nodes, while the
// Tanh rule (like Mul, Add and the other templated rules) is a model-local
// function that the gradient model inlines, so only Relu's gradient equals
// BuildBackward's nodes one for one.
std::string ReluText() {
  return Replace(ForwardText(), "T = Tanh(H)", "T = Relu(H)");
}

void TheOutputsAreTheWrtGradientsWithTheirShapes() {
  const onnx::ModelProto g =
      BuildGradientModel(Parse(ForwardText()), "S", {"W", "X", "U"}, "p_");
  Check(g.graph().output_size() == 3, "one output per wrt");
  const onnx::ValueInfoProto* dw = Output(g, "p_grad_0");
  const onnx::ValueInfoProto* dx = Output(g, "p_grad_1");
  const onnx::ValueInfoProto* du = Output(g, "p_grad_2");
  if (!(dw && dx && du)) {
    Check(false, "the outputs are named <prefix>grad_<k>");
    return;
  }
  Check(Dims(*dw) == std::vector<int64_t>({3, 4}),
        "the gradient of W is (3, 4)");
  Check(Dims(*dx) == std::vector<int64_t>({2, 3}),
        "the gradient of X is (2, 3)");
  Check(Dims(*du) == std::vector<int64_t>({5}), "the gradient of U is (5)");
  Check(dw->type().tensor_type().elem_type() == onnx::TensorProto::FLOAT,
        "float32 in, float32 out");
  Check(g.graph().node(0).op_type() == "MatMul" &&
            g.graph().node(0).output(0) == "H",
        "the forward nodes the loss needs come first");
  Check(Producer(g, "Y") == nullptr,
        "a forward node the loss does not need is dropped");
  Check(g.graph().input_size() == 3, "the forward inputs are kept");
  Check(g.functions_size() == 0, "no model-local function is left behind");
  for (const onnx::OperatorSetIdProto& o : g.opset_import()) {
    Check(o.domain().empty() && o.version() == 18, "the forward opset is kept");
  }
  for (int k = kForwardNodes; k < g.graph().node_size(); ++k) {
    for (const std::string& o : g.graph().node(k).output()) {
      Check(o.compare(0, 2, "p_") == 0,
            "a new tensor name starts with the prefix: " + o);
    }
  }
  for (const onnx::TensorProto& t : g.graph().initializer()) {
    Check(t.name().compare(0, 2, "p_") == 0,
          "a new initializer starts with the prefix: " + t.name());
  }
}

// The gradient nodes are exactly BuildBackward's nodes for the same slice and
// seed, with the same names: the whole-model entry point adds no arithmetic.
void TheGradientIsBuildBackwardOfTheSlice() {
  const onnx::ModelProto fwd = Parse(ReluText());
  const onnx::ModelProto g =
      BuildGradientModel(fwd, "S", {"W", "X", "U"}, "p_");

  std::map<std::string, std::vector<int64_t>> shapes = {
      {"X", {2, 3}}, {"W", {3, 4}}, {"H", {2, 4}}, {"T", {2, 4}}, {"S", {}}};
  // MatMul, Relu and ReduceSum: the forward nodes the loss needs (Y is not)
  std::vector<onnx::NodeProto> slice = {
      fwd.graph().node(0), fwd.graph().node(1), fwd.graph().node(3)};
  GraphBuilder b("p_");
  std::map<std::string, std::string> seeds = {{"S", b.Const(1.0f, "seed")}};
  const std::map<std::string, std::string> got =
      BuildBackward(b, slice, shapes, seeds, {"W", "X"});

  // then the zero for U (an Expand) and one Identity per wrt
  const int n = static_cast<int>(b.nodes().size());
  Check(g.graph().node_size() == kForwardNodes + n + 1 + 3,
        "the model is the forward nodes, BuildBackward's nodes, then the zero "
        "and the Identities");
  bool same = true;
  for (int k = 0; k < n && same; ++k) {
    same = g.graph().node(kForwardNodes + k).SerializeAsString() ==
           b.nodes()[k].SerializeAsString();
  }
  Check(same,
        "the gradient nodes equal BuildBackward's nodes for the same slice");
  Check(got.count("W") == 1 && got.count("X") == 1,
        "BuildBackward reaches W and X");
  Check(Producer(g, "p_grad_0") != nullptr &&
            Producer(g, "p_grad_0")->op_type() == "Identity" &&
            Producer(g, "p_grad_0")->input(0) == got.at("W"),
        "p_grad_0 is an Identity of BuildBackward's gradient of W");
  Check(Producer(g, "p_grad_1") != nullptr &&
            Producer(g, "p_grad_1")->input(0) == got.at("X"),
        "p_grad_1 is an Identity of BuildBackward's gradient of X");
}

// U, which the loss does not use, gets a zero of its shape: Identity of an
// Expand of a scalar zero to (5).
void AnUnreachedTargetGetsZeros() {
  const onnx::ModelProto g =
      BuildGradientModel(Parse(ForwardText()), "S", {"U"}, "z_");
  const onnx::NodeProto* id = Producer(g, "z_grad_0");
  Check(id != nullptr && id->op_type() == "Identity",
        "the zero gradient is an Identity");
  if (id == nullptr) return;
  const onnx::NodeProto* expand = Producer(g, id->input(0));
  Check(expand != nullptr && expand->op_type() == "Expand",
        "the zero is expanded to the shape of U");
  if (expand == nullptr) return;
  const onnx::TensorProto* shape = Initializer(g, expand->input(1));
  Check(shape != nullptr && Int64Values(*shape) == std::vector<int64_t>({5}),
        "the expand target is the shape (5) of U");
  const onnx::TensorProto* zero = Initializer(g, expand->input(0));
  Check(zero != nullptr && zero->data_type() == onnx::TensorProto::FLOAT,
        "the zero is a float32 scalar");
  Check(zero != nullptr && zero->dims_size() == 0, "the zero is a scalar");
  Check(zero != nullptr && FloatValues(*zero) == std::vector<float>({0.0f}),
        "the zero is 0");
}

// Only the wrt the loss reaches have their gradient computed; the rest are
// zero.
void ALossThatIsAWrtGetsTheSeed() {
  const onnx::ModelProto g =
      BuildGradientModel(Parse(ForwardText()), "S", {"S"}, "s_");
  Check(g.graph().output_size() == 1, "one output for the one wrt");
  if (g.graph().output_size() == 1) {
    Check(Dims(g.graph().output(0)).empty(), "d S / d S is a scalar");
  }
}

// Tanh's rule is a model-local function: the gradient model holds only
// standard operators, with no function and no custom domain left.
void TheTemplatedRulesAreInlined() {
  const onnx::ModelProto g =
      BuildGradientModel(Parse(ForwardText()), "S", {"X"}, "t_");
  Check(g.functions_size() == 0,
        "no model-local function is left in the gradient model");
  bool has_mul = false;
  for (const onnx::NodeProto& nd : g.graph().node()) {
    Check(nd.domain().empty(),
          "a node of a custom domain is left: " + nd.op_type());
    has_mul = has_mul || nd.op_type() == "Mul";
  }
  Check(has_mul, "the inlined Tanh rule computes with Mul");
}

void AFloat64ModelGetsFloat64Gradients() {
  const onnx::ModelProto g =
      BuildGradientModel(Parse(ForwardText("double")), "S", {"W"}, "d_");
  Check(Output(g, "d_grad_0") != nullptr &&
            Output(g, "d_grad_0")->type().tensor_type().elem_type() ==
                onnx::TensorProto::DOUBLE,
        "float64 in, float64 out");
  for (const onnx::TensorProto& t : g.graph().initializer()) {
    if (t.name().compare(0, 2, "d_") != 0 ||
        t.data_type() == onnx::TensorProto::INT64)
      continue;
    Check(t.data_type() == onnx::TensorProto::DOUBLE,
          "a float64 constant: " + t.name());
  }
}

void AnOperatorWithoutARuleIsRefused() {
  const std::string text = Replace(ReluText(), "T = Relu(H)", "T = Sin(H)");
  bool threw = false;
  try {
    BuildGradientModel(Parse(text), "S", {"X"}, "q_");
  } catch (const UnsupportedOpError& e) {
    threw = std::string(e.what()).find("Sin") != std::string::npos;
  }
  Check(threw, "UnsupportedOpError names the operator");
}

void ABadLossOrWrtIsRefused() {
  const onnx::ModelProto fwd = Parse(ForwardText());
  auto refused = [&](const std::string& loss,
                     const std::vector<std::string>& wrt,
                     const std::string& fragment) {
    try {
      BuildGradientModel(fwd, loss, wrt, "q_");
    } catch (const std::invalid_argument& e) {
      return std::string(e.what()).find(fragment) != std::string::npos;
    }
    return false;
  };
  Check(refused("H", {"X"}, "must be a scalar"),
        "a loss with a shape is refused");
  Check(refused("nope", {"X"}, "not a value"), "an unknown loss is refused");
  Check(refused("S", {"nope"}, "not a value"), "an unknown wrt is refused");
}

// d/dX of d S / d W, where S = ReduceSum(Tanh(X @ W)): the gradient model is a
// model like any other, so it can be differentiated again.
void ASecondDerivativeWorks() {
  onnx::ModelProto first =
      BuildGradientModel(Parse(ForwardText()), "S", {"W"}, "a_");
  onnx::NodeProto* sum = first.mutable_graph()->add_node();
  sum->set_op_type("ReduceSum");
  sum->add_input("a_grad_0");
  sum->add_output("L2");
  onnx::AttributeProto* keep = sum->add_attribute();
  keep->set_name("keepdims");
  keep->set_type(onnx::AttributeProto::INT);
  keep->set_i(0);
  const onnx::ModelProto second = BuildGradientModel(first, "L2", {"X"}, "b_");
  Check(second.graph().output_size() == 1,
        "one output for the second derivative");
  if (second.graph().output_size() == 1) {
    Check(Dims(second.graph().output(0)) == std::vector<int64_t>({2, 3}),
          "d L2 / d X has the shape of X");
  }
}

// Two calls with the same prefix give the same model; with different prefixes
// the new names are disjoint, so the two gradient models merge without a clash.
void RepeatedCallsDoNotCollide() {
  const onnx::ModelProto fwd = Parse(ForwardText());
  const onnx::ModelProto a1 = BuildGradientModel(fwd, "S", {"W", "X"}, "a_");
  const onnx::ModelProto a2 = BuildGradientModel(fwd, "S", {"W", "X"}, "a_");
  Check(a1.SerializeAsString() == a2.SerializeAsString(),
        "the same call twice gives the same model");

  const onnx::ModelProto b = BuildGradientModel(fwd, "S", {"W", "X"}, "b_");
  std::set<std::string> outputs_a;
  std::set<std::string> outputs_b;
  for (int k = kForwardNodes; k < a1.graph().node_size(); ++k) {
    for (const std::string& o : a1.graph().node(k).output())
      outputs_a.insert(o);
  }
  for (int k = kForwardNodes; k < b.graph().node_size(); ++k) {
    for (const std::string& o : b.graph().node(k).output()) outputs_b.insert(o);
  }
  bool disjoint = true;
  for (const std::string& o : outputs_b)
    disjoint = disjoint && outputs_a.count(o) == 0;
  Check(disjoint, "two prefixes give disjoint gradient names");
  Check(!outputs_a.empty() && !outputs_b.empty(),
        "both models have gradient nodes");
}

}  // namespace

int main() {
  TheOutputsAreTheWrtGradientsWithTheirShapes();
  TheGradientIsBuildBackwardOfTheSlice();
  AnUnreachedTargetGetsZeros();
  TheTemplatedRulesAreInlined();
  ALossThatIsAWrtGetsTheSeed();
  AFloat64ModelGetsFloat64Gradients();
  AnOperatorWithoutARuleIsRefused();
  ABadLossOrWrtIsRefused();
  ASecondDerivativeWorks();
  RepeatedCallsDoNotCollide();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::fprintf(stderr, "all graph_grad_model checks passed\n");
  return 0;
}
