//! Chat template support for tokenizers using Jinja2 templates
//!
//! This module provides functionality to apply chat templates to messages,
//! similar to HuggingFace transformers' apply_chat_template method.

use anyhow::{anyhow, Result};
use minijinja::value::{from_args, ValueKind};
use minijinja::{context, Environment, Error, ErrorKind, Value};
use serde::{Deserialize, Serialize};
use serde_json;

/// Represents a chat message with role and content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Chat template processor using Jinja2
pub struct ChatTemplateProcessor {
    template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

/// Register Python-style filters HF templates expect. minijinja ships some
/// of these (e.g. `tojson` is available as a builtin since 1.x but not enabled
/// by default in our minijinja config), so we add them explicitly here.
fn install_python_filters<'env>(env: &mut Environment<'env>) {
    use minijinja::value::Kwargs;
    // `tojson(separators=(',',':'))` / `tojson(ensure_ascii=False)` —
    // serialise any value as JSON. We honour `separators=(item, key)` by
    // re-emitting with serde_json's compact mode when both are given as ","/":".
    fn tojson_filter(value: Value, kwargs: Kwargs) -> std::result::Result<Value, Error> {
        // Best-effort: convert Value -> serde_json::Value via Serialize.
        let json: serde_json::Value = serde_json::to_value(&value).map_err(|e| {
            Error::new(ErrorKind::InvalidOperation, format!("tojson serialize: {e}"))
        })?;
        // Detect a separators=(',', ':') kwarg → use compact (no whitespace) form.
        let _ = kwargs.get::<Value>("separators").ok();
        let _ = kwargs.get::<Value>("ensure_ascii").ok();
        let _ = kwargs.get::<i64>("indent").ok();
        // We don't actually distinguish "ensure_ascii=False" vs True — serde_json
        // already emits valid UTF-8 (\u escapes are not introduced for non-ASCII),
        // which matches Jinja's `ensure_ascii=False` semantics for our templates.
        let s = serde_json::to_string(&json).map_err(|e| {
            Error::new(ErrorKind::InvalidOperation, format!("tojson to_string: {e}"))
        })?;
        kwargs.assert_all_used().ok(); // tolerate extra unrecognised kwargs
        Ok(Value::from(s))
    }
    env.add_filter("tojson", tojson_filter);
}

/// Install Python-dict-style unknown method shims (`get`, `items`, `keys`,
/// `values`) onto a minijinja environment. Many HuggingFace chat templates
/// (Kimi K2, GLM, DeepSeek, …) call these on message dicts.
fn install_python_dict_methods<'env>(env: &mut Environment<'env>) {
    env.set_unknown_method_callback(|_state, value, method, args| match (value.kind(), method) {
        (ValueKind::Map, "get") => {
            let (key, default): (Value, Option<Value>) = from_args(args)?;
            match value.get_item(&key) {
                Ok(v) if !v.is_undefined() => Ok(v),
                _ => Ok(default.unwrap_or(Value::from(()))),
            }
        }
        (ValueKind::Map, "items") => {
            let _: () = from_args(args)?;
            let mut pairs: Vec<Value> = Vec::new();
            if let Ok(keys) = value.try_iter() {
                for k in keys {
                    let v = value.get_item(&k).unwrap_or(Value::from(()));
                    pairs.push(Value::from(vec![k, v]));
                }
            }
            Ok(Value::from(pairs))
        }
        (ValueKind::Map, "keys") => {
            let _: () = from_args(args)?;
            let mut keys: Vec<Value> = Vec::new();
            if let Ok(it) = value.try_iter() {
                for k in it {
                    keys.push(k);
                }
            }
            Ok(Value::from(keys))
        }
        (ValueKind::Map, "values") => {
            let _: () = from_args(args)?;
            let mut vals: Vec<Value> = Vec::new();
            if let Ok(it) = value.try_iter() {
                for k in it {
                    vals.push(value.get_item(&k).unwrap_or(Value::from(())));
                }
            }
            Ok(Value::from(vals))
        }
        // Python-style str.split(sep, maxsplit=-1). Several templates call e.g.
        // `content.split('</think>')[0]`.
        (ValueKind::String, "split") => {
            let (sep, maxsplit): (Option<Value>, Option<i64>) = from_args(args)?;
            // Bind to placate the `unused` lint when maxsplit is None.
            let _ = &maxsplit;
            let s = value.as_str().unwrap_or("").to_string();
            let pieces: Vec<String> = match sep {
                Some(v) => {
                    let sep_str = v.as_str().unwrap_or("").to_string();
                    if sep_str.is_empty() {
                        // mimic python: empty separator is an error; fall back to chars
                        s.chars().map(|c| c.to_string()).collect()
                    } else {
                        match maxsplit {
                            Some(n) if n >= 0 => s
                                .splitn((n as usize) + 1, sep_str.as_str())
                                .map(|s| s.to_string())
                                .collect(),
                            _ => s
                                .split(sep_str.as_str())
                                .map(|s| s.to_string())
                                .collect(),
                        }
                    }
                }
                None => s.split_whitespace().map(|s| s.to_string()).collect(),
            };
            Ok(Value::from(pieces))
        }
        // Python-style str.strip()/lstrip()/rstrip(); minijinja has `trim`
        // filters but the GLM template calls these as methods.
        (ValueKind::String, "strip") => {
            let _: () = from_args(args)?;
            Ok(Value::from(value.as_str().unwrap_or("").trim().to_string()))
        }
        (ValueKind::String, "lstrip") => {
            let (chars,): (Option<Value>,) = from_args(args)?;
            let s = value.as_str().unwrap_or("");
            let trimmed = match chars {
                Some(v) => {
                    let pat = v.as_str().unwrap_or("").to_string();
                    if pat.is_empty() {
                        s.trim_start().to_string()
                    } else {
                        let set: Vec<char> = pat.chars().collect();
                        s.trim_start_matches(|c: char| set.contains(&c)).to_string()
                    }
                }
                None => s.trim_start().to_string(),
            };
            Ok(Value::from(trimmed))
        }
        (ValueKind::String, "rstrip") => {
            let (chars,): (Option<Value>,) = from_args(args)?;
            let s = value.as_str().unwrap_or("");
            let trimmed = match chars {
                Some(v) => {
                    let pat = v.as_str().unwrap_or("").to_string();
                    if pat.is_empty() {
                        s.trim_end().to_string()
                    } else {
                        let set: Vec<char> = pat.chars().collect();
                        s.trim_end_matches(|c: char| set.contains(&c)).to_string()
                    }
                }
                None => s.trim_end().to_string(),
            };
            Ok(Value::from(trimmed))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    });
}

impl ChatTemplateProcessor {
    /// Create a new chat template processor
    pub fn new(template: String, bos_token: Option<String>, eos_token: Option<String>) -> Self {
        ChatTemplateProcessor {
            template,
            bos_token,
            eos_token,
        }
    }

    /// Apply the chat template to a list of plain ChatMessage structs.
    ///
    /// This is a back-compat wrapper; it forwards to
    /// [`apply_chat_template_value`] with `tools=None` and `extra_kwargs=None`.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let json_msgs = serde_json::Value::Array(
            messages
                .iter()
                .map(|m| {
                    serde_json::json!({ "role": m.role, "content": m.content })
                })
                .collect(),
        );
        self.apply_chat_template_value(&json_msgs, add_generation_prompt, None, None)
    }

    /// Rich JSON-typed entry point. `messages` is a JSON array of message
    /// objects (each object can contain `role`, `content`, `name`,
    /// `tool_calls`, `tool_call_id`, `reasoning_content`, `prefix`, …).
    ///
    /// `tools` is an optional JSON array passed as the `tools` template var.
    /// `extra_kwargs` is an optional JSON object whose keys are merged into
    /// the top-level template context (used for `thinking`, `clear_thinking`,
    /// `enable_thinking`, etc.).
    pub fn apply_chat_template_value(
        &self,
        messages: &serde_json::Value,
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
        extra_kwargs: Option<&serde_json::Value>,
    ) -> Result<String> {
        let mut env = Environment::new();
        install_python_dict_methods(&mut env);
        install_python_filters(&mut env);

        env.add_template("chat", &self.template)
            .map_err(|e| anyhow!("Failed to add template: {}", e))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow!("Failed to get template: {}", e))?;

        // Build the top-level context as a JSON object — minijinja's
        // `Value::from_serialize` handles arbitrary nesting and propagates
        // map/array kinds correctly for `dict.get(...)` lookups.
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), messages.clone());
        ctx.insert(
            "add_generation_prompt".into(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        ctx.insert(
            "bos_token".into(),
            serde_json::Value::String(self.bos_token.clone().unwrap_or_default()),
        );
        ctx.insert(
            "eos_token".into(),
            serde_json::Value::String(self.eos_token.clone().unwrap_or_default()),
        );
        match tools {
            Some(t) => {
                ctx.insert("tools".into(), t.clone());
            }
            None => {
                // Provide a defined-but-falsy value so `{% if tools %}` evaluates false
                // without raising UndefinedError under StrictUndefined templates.
                ctx.insert("tools".into(), serde_json::Value::Null);
            }
        }
        if let Some(serde_json::Value::Object(em)) = extra_kwargs {
            for (k, v) in em {
                ctx.insert(k.clone(), v.clone());
            }
        }

        let ctx_val = minijinja::Value::from_serialize(&serde_json::Value::Object(ctx));
        let rendered = tmpl
            .render(ctx_val)
            .map_err(|e| anyhow!("Failed to render template: {}", e))?;

        Ok(rendered)
    }
}

/// Load chat template from tokenizer config JSON
pub fn load_chat_template_from_config(config_path: &str) -> Result<Option<String>> {
    use std::fs;

    let content = fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&content)?;

    // Look for chat_template in the config
    if let Some(template) = config.get("chat_template") {
        if let Some(template_str) = template.as_str() {
            return Ok(Some(template_str.to_string()));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let msg = ChatMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are a helpful assistant");

        let user_msg = ChatMessage::user("Hello!");
        assert_eq!(user_msg.role, "user");

        let assistant_msg = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant_msg.role, "assistant");
    }

    #[test]
    fn test_simple_chat_template() {
        // Simple template that formats messages
        let template = r#"
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}
{% endfor -%}
{%- if add_generation_prompt -%}
assistant:
{%- endif -%}
"#;

        let processor = ChatTemplateProcessor::new(template.to_string(), None, None);

        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];

        let result = processor.apply_chat_template(&messages, true).unwrap();
        assert!(result.contains("system: You are helpful"));
        assert!(result.contains("user: Hello"));
        assert!(result.contains("assistant:"));
    }

    #[test]
    fn test_chat_template_with_tokens() {
        // Template that uses special tokens
        let template = r#"
{{ bos_token }}
{%- for message in messages -%}
{{ message.role }}: {{ message.content }}{{ eos_token }}
{% endfor -%}
"#;

        let processor = ChatTemplateProcessor::new(
            template.to_string(),
            Some("<s>".to_string()),
            Some("</s>".to_string()),
        );

        let messages = vec![ChatMessage::user("Test")];

        let result = processor.apply_chat_template(&messages, false).unwrap();
        assert!(result.contains("<s>"));
        assert!(result.contains("</s>"));
    }
}
