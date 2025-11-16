#!/usr/bin/env python3
"""
使用外部tokenizer的训练脚本
支持从Hugging Face Hub加载预训练tokenizer，避免本地训练
"""

import os
import sys
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def setup_external_tokenizer(tokenizer_name="Qwen/Qwen2.5-0.5B", save_dir="../model/external_tokenizer"):
    """
    设置外部tokenizer
    
    Args:
        tokenizer_name: Hugging Face模型名称
        save_dir: 保存tokenizer文件的目录
    
    Returns:
        tokenizer: 配置好的tokenizer对象
        vocab_size: 词汇表大小
    """
    
    print(f"🚀 正在设置外部tokenizer: {tokenizer_name}")
    
    try:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 从Hugging Face加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            local_files_only=False
        )
        
        print(f"✅ 成功加载tokenizer: {tokenizer_name}")
        print(f"   词汇表大小: {tokenizer.vocab_size}")
        
        # 适配MiniMind的特殊token配置
        tokenizer = adapt_tokenizer_for_minimind(tokenizer)
        
        # 保存tokenizer文件
        tokenizer.save_pretrained(save_dir)
        print(f"✅ tokenizer文件已保存到: {save_dir}")
        
        # 创建tokenizer_config.json（适配MiniMind配置）
        create_minimind_tokenizer_config(tokenizer, save_dir)
        
        return tokenizer, tokenizer.vocab_size
        
    except Exception as e:
        print(f"❌ 设置外部tokenizer失败: {e}")
        return None, None


def adapt_tokenizer_for_minimind(tokenizer):
    """适配tokenizer以兼容MiniMind项目的特殊token配置"""
    
    print("🔄 适配tokenizer以兼容MiniMind配置...")
    
    # MiniMind项目的特殊token
    minimind_tokens = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>"
    }
    
    # 检查并添加特殊token
    for token_name, token_value in minimind_tokens.items():
        current_token = getattr(tokenizer, token_name, None)
        
        if current_token != token_value:
            # 如果token不在词汇表中，需要添加
            if tokenizer.convert_tokens_to_ids(token_value) == tokenizer.unk_token_id:
                # 添加新token
                tokenizer.add_tokens([token_value], special_tokens=True)
                print(f"   添加特殊token: {token_value}")
            
            # 更新tokenizer配置
            setattr(tokenizer, token_name, token_value)
            setattr(tokenizer, f"{token_name}_id", tokenizer.convert_tokens_to_ids(token_value))
            print(f"   设置{token_name}: {token_value} (ID: {getattr(tokenizer, f'{token_name}_id')})")
    
    return tokenizer


def create_minimind_tokenizer_config(tokenizer, save_dir):
    """创建适配MiniMind的tokenizer配置文件"""
    
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": getattr(tokenizer, 'bos_token', '<|im_start|>'),
        "eos_token": getattr(tokenizer, 'eos_token', '<|im_end|>'),
        "pad_token": getattr(tokenizer, 'pad_token', '<|endoftext|>'),
        "unk_token": getattr(tokenizer, 'unk_token', '<|endoftext|>'),
        "model_max_length": 32768,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
 {%- if messages[0]['role'] == 'system' -%}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else -%}
        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}
 {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
   {{- '<|im_start|>' + message.role + '\\n' + content }}
  {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\"name\": \"' }}
                {{- tool_call.name }}
                {{- '\", \"arguments\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '🛠️\\n\\n🔧\\n\\n' }}
    {%- endif %}
{%- endif %}"""
    }
    
    import json
    config_path = os.path.join(save_dir, "tokenizer_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 创建tokenizer配置文件: {config_path}")


def validate_tokenizer(tokenizer, test_texts=None):
    """验证tokenizer功能"""
    
    if test_texts is None:
        test_texts = [
            "你好，这是一个测试",
            "Hello, this is a test",
            "机器学习模型训练",
            "Natural language processing"
        ]
    
    print("\n🧪 验证tokenizer功能:")
    
    for text in test_texts:
        # 编码测试
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        
        print(f"  文本: {text[:30]}...")
        print(f"  Token数量: {len(encoded)}")
        print(f"  解码结果: {decoded[:30]}...")
        
        # 检查特殊token
        if hasattr(tokenizer, 'bos_token_id'):
            print(f"  BOS Token ID: {tokenizer.bos_token_id}")
        if hasattr(tokenizer, 'eos_token_id'):
            print(f"  EOS Token ID: {tokenizer.eos_token_id}")
        
        print("  ---")


def main():
    """主函数"""
    
    print("🚀 MiniMind项目 - 使用外部tokenizer训练")
    print("=" * 50)
    
    # 可选的tokenizer列表
    tokenizer_options = [
        "Qwen/Qwen2.5-0.5B",      # 推荐：中文优化，词汇表合理
        "Qwen/Qwen2.5-1.5B",      # 中等规模
        "THUDM/chatglm3-6b",      # ChatGLM tokenizer
        "baichuan-inc/Baichuan2-7B-Base",  # Baichuan tokenizer
        "meta-llama/Llama-3.2-1B" # Llama tokenizer（英文优化）
    ]
    
    # 选择第一个选项（Qwen 0.5B，推荐用于中文）
    selected_tokenizer = tokenizer_options[0]
    
    # 设置外部tokenizer
    tokenizer, vocab_size = setup_external_tokenizer(selected_tokenizer)
    
    if tokenizer is None:
        print("❌ tokenizer设置失败，退出程序")
        return
    
    # 验证tokenizer
    validate_tokenizer(tokenizer)
    
    print("\n✅ 外部tokenizer设置完成！")
    print("\n📋 下一步操作:")
    print("1. 修改模型配置中的vocab_size为: {}".format(vocab_size))
    print("2. 在训练脚本中使用新的tokenizer路径")
    print("3. 开始模型训练")
    print("\n💡 提示: 外部tokenizer已经过大规模数据训练，通常比本地训练的效果更好")


if __name__ == "__main__":
    main()