// C++ gRPC client for the tokenizer service. Mirrors the Rust client: drives
// every RPC with the deterministic `gpt-3.5-turbo` tiktoken backend (no
// network), plus the built-in "mock" tokenizer, and asserts roundtrip equality.

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "tokenizer.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::Status;

using tokenizer::v1::ChatMessage;
using tokenizer::v1::DetokenizeChunk;
using tokenizer::v1::DetokenizeRequest;
using tokenizer::v1::DetokenizeResponse;
using tokenizer::v1::DetokenizeStreamRequest;
using tokenizer::v1::GetModelInfoRequest;
using tokenizer::v1::GetModelInfoResponse;
using tokenizer::v1::TokenizeRequest;
using tokenizer::v1::TokenizeResponse;
using tokenizer::v1::Tokenizer;

namespace {

void Die(const std::string& msg) {
  std::cerr << "[cpp_client] FAIL: " << msg << std::endl;
  std::exit(1);
}

void Check(bool cond, const std::string& msg) {
  if (!cond) Die(msg);
}

void CheckStatus(const Status& s, const std::string& where) {
  if (!s.ok()) {
    Die(where + ": " + std::to_string(static_cast<int>(s.error_code())) + " " +
        s.error_message());
  }
}

std::string VecToStr(const std::vector<uint32_t>& v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(v[i]);
  }
  out += "]";
  return out;
}

class TokenizerClient {
 public:
  explicit TokenizerClient(std::shared_ptr<Channel> ch)
      : stub_(Tokenizer::NewStub(ch)) {}

  GetModelInfoResponse GetInfo(const std::string& model) {
    GetModelInfoRequest req;
    req.set_model(model);
    GetModelInfoResponse resp;
    ClientContext ctx;
    auto s = stub_->GetModelInfo(&ctx, req, &resp);
    CheckStatus(s, "GetModelInfo");
    return resp;
  }

  TokenizeResponse TokenizeRaw(const std::string& model,
                               const std::string& text) {
    TokenizeRequest req;
    req.set_model(model);
    req.set_raw_text(text);
    TokenizeResponse resp;
    ClientContext ctx;
    auto s = stub_->Tokenize(&ctx, req, &resp);
    CheckStatus(s, "Tokenize(raw)");
    return resp;
  }

  TokenizeResponse TokenizeChat(const std::string& model,
                                const std::vector<std::pair<std::string, std::string>>& msgs,
                                bool add_generation_prompt) {
    TokenizeRequest req;
    req.set_model(model);
    req.set_add_generation_prompt(add_generation_prompt);
    for (const auto& [role, content] : msgs) {
      auto* m = req.add_messages();
      m->set_role(role);
      m->set_content(content);
    }
    TokenizeResponse resp;
    ClientContext ctx;
    auto s = stub_->Tokenize(&ctx, req, &resp);
    CheckStatus(s, "Tokenize(chat)");
    return resp;
  }

  std::string Detokenize(const std::string& model,
                         const std::vector<uint32_t>& token_ids,
                         bool skip_special_tokens) {
    DetokenizeRequest req;
    req.set_model(model);
    req.set_skip_special_tokens(skip_special_tokens);
    for (auto id : token_ids) req.add_token_ids(id);
    DetokenizeResponse resp;
    ClientContext ctx;
    auto s = stub_->Detokenize(&ctx, req, &resp);
    CheckStatus(s, "Detokenize");
    return resp.text();
  }

  std::string DetokenizeStream(const std::string& model,
                               const std::vector<uint32_t>& token_ids,
                               bool skip_special_tokens,
                               int* chunks_out) {
    DetokenizeStreamRequest req;
    req.set_model(model);
    req.set_skip_special_tokens(skip_special_tokens);
    for (auto id : token_ids) req.add_token_ids(id);

    ClientContext ctx;
    std::unique_ptr<ClientReader<DetokenizeChunk>> reader(
        stub_->DetokenizeStream(&ctx, req));

    std::string acc;
    int chunks = 0;
    DetokenizeChunk chunk;
    while (reader->Read(&chunk)) {
      ++chunks;
      acc += chunk.delta();
      std::cout << "[stream chunk] token_index=" << chunk.token_index()
                << " delta=" << chunk.delta() << "\n";
    }
    auto s = reader->Finish();
    CheckStatus(s, "DetokenizeStream");
    if (chunks_out) *chunks_out = chunks;
    return acc;
  }

 private:
  std::unique_ptr<Tokenizer::Stub> stub_;
};

}  // namespace

int main(int argc, char** argv) {
  std::string addr = "127.0.0.1:50051";
  if (argc > 1) addr = argv[1];
  if (const char* env = std::getenv("TOKENIZER_ADDR")) addr = env;

  std::cout << "[cpp_client] connecting to " << addr << std::endl;

  auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
  TokenizerClient client(channel);

  // ---- GetModelInfo (tiktoken) ----
  auto info = client.GetInfo("gpt-3.5-turbo");
  std::cout << "[GetModelInfo] model=" << info.model()
            << " vocab_size=" << info.vocab_size()
            << " bos=" << info.bos_token() << " eos=" << info.eos_token()
            << " has_chat_template=" << info.has_chat_template() << "\n";
  Check(info.vocab_size() > 0, "vocab_size should be > 0");

  // ---- Tokenize chat ----
  auto chat = client.TokenizeChat("gpt-3.5-turbo",
                                  {{"user", "Hello"}}, /*add_gen_prompt=*/true);
  std::cout << "[Tokenize chat] prompt=" << chat.prompt() << " num_tokens="
            << chat.num_tokens() << "\n";

  // ---- Tokenize raw ----
  auto raw = client.TokenizeRaw("gpt-3.5-turbo", "Hello, world!");
  std::vector<uint32_t> ids(raw.token_ids().begin(), raw.token_ids().end());
  std::cout << "[Tokenize raw] prompt=" << raw.prompt()
            << " num_tokens=" << raw.num_tokens()
            << " token_ids=" << VecToStr(ids) << "\n";
  Check(!ids.empty(), "tiktoken should produce non-empty ids");

  // ---- Detokenize roundtrip ----
  std::string text = client.Detokenize("gpt-3.5-turbo", ids,
                                       /*skip_special_tokens=*/true);
  std::cout << "[Detokenize] text=" << text << "\n";
  Check(text == "Hello, world!",
        "roundtrip mismatch: got '" + text + "' want 'Hello, world!'");

  // ---- DetokenizeStream ----
  int chunks = 0;
  std::string streamed = client.DetokenizeStream(
      "gpt-3.5-turbo", ids, /*skip_special_tokens=*/true, &chunks);
  std::cout << "[DetokenizeStream] chunks=" << chunks
            << " joined=" << streamed << "\n";
  Check(streamed == text, "stream concat must equal unary decode");

  // ---- Mock tokenizer (no network) ----
  auto mock = client.TokenizeRaw("mock", "hello world");
  std::vector<uint32_t> mock_ids(mock.token_ids().begin(),
                                 mock.token_ids().end());
  std::cout << "[Tokenize mock] num_tokens=" << mock.num_tokens()
            << " token_ids=" << VecToStr(mock_ids) << "\n";
  Check(!mock_ids.empty(), "mock tokenizer should produce non-empty ids");

  std::string mock_text = client.Detokenize("mock", mock_ids, true);
  std::cout << "[Detokenize mock] text=" << mock_text << "\n";
  Check(!mock_text.empty(), "mock detokenize should produce non-empty text");

  // ---- Optional Kimi tokenize (only if KIMI_MODEL is set) ----
  if (const char* kimi_model = std::getenv("KIMI_MODEL")) {
    std::string text = "Hello, Kimi! 你好。";
    if (const char* kt = std::getenv("KIMI_TEXT")) text = kt;
    auto kresp = client.TokenizeRaw(kimi_model, text);
    std::vector<uint32_t> kids(kresp.token_ids().begin(),
                               kresp.token_ids().end());
    std::cout << "[Tokenize kimi raw] model=" << kimi_model << " text=" << text
              << " num_tokens=" << kresp.num_tokens()
              << " token_ids=" << VecToStr(kids) << "\n";
  }

  std::cout << "C++ client tests passed" << std::endl;
  return 0;
}
