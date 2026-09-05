import requests
import json


def get_weather(city: str, days: int = 3) -> str:
    """
    通过调用 wttr.in API 查询指定城市未来几天的天气信息。

    Args:
        city: 城市名称，中文或英文均可
        days: 查询天数，默认3天

    Returns:
        格式化的天气信息字符串
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        weather_desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]

        result = [
            f"【{city}当前天气】",
            f"天气状况: {weather_desc}",
            f"气温: {temp_c}°C (体感 {feels_like}°C)",
            f"湿度: {humidity}%",
        ]

        # 获取未来几天的预报
        forecasts = data.get("weather", [])
        if forecasts:
            result.append(f"\n【未来{min(days, len(forecasts))}天预报】")
            for i, day in enumerate(forecasts[:days]):
                date = day["date"]
                max_temp = day["maxtempC"]
                min_temp = day["mintempC"]
                hourly = day.get("hourly", [{}])
                desc = hourly[0].get("weatherDesc", [{}])[0].get("value", "未知")
                result.append(
                    f"  {date}: {desc}, 最高{max_temp}°C, 最低{min_temp}°C"
                )

        return "\n".join(result)

    except requests.exceptions.RequestException as e:
        return f"天气查询失败(网络错误): {e}"
    except (KeyError, IndexError) as e:
        return f"天气数据解析失败: {e}"
    except Exception as e:
        return f"天气查询未知错误: {e}"


if __name__ == "__main__":
    print(get_weather("北京", days=3))