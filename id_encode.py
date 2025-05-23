def category_id_to_255bit(category_id: str) -> str:
    """
    将字符串 category_id（十进制整数）转换为长度固定为255的二进制字符串，
    高位不足时左侧补0。

    Args:
        category_id: 形如 "12345" 的十进制数值字符串。

    Returns:
        长度恰好为255的二进制字符串（只包含 '0' 和 '1'）。
    
    Raises:
        ValueError: 如果 category_id 不是合法整数，或其二进制表示已经超过255位。
    """
    # 1. 转成整数
    try:
        n = int(category_id)
    except ValueError:
        raise ValueError(f"无法将 '{category_id}' 转为整数")

    if n < 0:
        raise ValueError("category_id 必须是非负整数")

    # 2. 得到不带 '0b' 前缀的二进制表示
    bits = format(n, 'b')  # 或者 bin(n)[2:]

    # 3. 检查长度，不得超过255
    if len(bits) > 255:
        raise ValueError(
            f"二进制位数 ({len(bits)}) 已超过255，无法在255位内表示"
        )

    # 4. 左侧补0到255位
    return bits.zfill(255)


# 示例
if __name__ == "__main__":
    for cid in ["13", "0", "65535"]:
        b255 = category_id_to_255bit(cid)
        print(f"{cid} => (len={len(b255)}) {b255[:16]}...{b255[-16:]}")
