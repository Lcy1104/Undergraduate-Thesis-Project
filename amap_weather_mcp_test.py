import requests
import json
from typing import Dict, Optional


class GaodeWeatherAPI:
    def __init__(self, api_key: str):
        """
        初始化高德天气API客户端

        Args:
            api_key: 高德开放平台申请的Web服务API Key
        """
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3/weather/weatherInfo"

    def get_weather(self, city_code: str, extensions: str = "base", output: str = "JSON") -> Optional[Dict]:
        """
        获取天气信息

        Args:
            city_code: 城市adcode编码
            extensions: 气象类型，可选值：base/all
                       base: 返回实况天气
                       all: 返回预报天气
            output: 返回格式，可选值：JSON/XML，默认为JSON

        Returns:
            返回天气数据的字典，失败返回None
        """
        params = {
            "key": self.api_key,
            "city": city_code,
            "extensions": extensions,
            "output": output
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # 检查API返回状态
            if data.get("status") == "1" and data.get("infocode") == "10000":
                return data
            else:
                print(f"API错误: {data.get('info', '未知错误')}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"网络请求错误: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None

    def parse_current_weather(self, data: Dict) -> Optional[Dict]:
        """
        解析实况天气数据

        Args:
            data: API返回的原始数据

        Returns:
            解析后的实况天气信息字典
        """
        if not data or "lives" not in data or not data["lives"]:
            return None

        live_data = data["lives"][0]
        return {
            "province": live_data.get("province"),
            "city": live_data.get("city"),
            "adcode": live_data.get("adcode"),
            "weather": live_data.get("weather"),
            "temperature": live_data.get("temperature"),
            "wind_direction": live_data.get("winddirection"),
            "wind_power": live_data.get("windpower"),
            "humidity": live_data.get("humidity"),
            "report_time": live_data.get("reporttime")
        }

    def parse_forecast_weather(self, data: Dict) -> Optional[Dict]:
        """
        解析预报天气数据

        Args:
            data: API返回的原始数据

        Returns:
            解析后的预报天气信息字典
        """
        if not data or "forecasts" not in data or not data["forecasts"]:
            return None

        forecast_data = data["forecasts"][0]
        casts = forecast_data.get("casts", [])

        forecast_list = []
        for cast in casts:
            forecast_list.append({
                "date": cast.get("date"),
                "week": cast.get("week"),
                "day_weather": cast.get("dayweather"),
                "night_weather": cast.get("nightweather"),
                "day_temp": cast.get("daytemp"),
                "night_temp": cast.get("nighttemp"),
                "day_wind": cast.get("daywind"),
                "night_wind": cast.get("nightwind"),
                "day_power": cast.get("daypower"),
                "night_power": cast.get("nightpower")
            })

        return {
            "city": forecast_data.get("city"),
            "adcode": forecast_data.get("adcode"),
            "province": forecast_data.get("province"),
            "report_time": forecast_data.get("reporttime"),
            "forecasts": forecast_list
        }


def main():
    # 使用示例
    API_KEY = "16a98a1a06e46becd9a8689f18074882"  # 请替换为你的实际API Key

    # 常见城市adcode编码
    city_codes = {
        "北京": "110101",
        "上海": "310101",
        "广州": "440103",
        "深圳": "440303",
        "杭州": "330102",
        "成都": "510104"
    }

    weather_api = GaodeWeatherAPI(API_KEY)

    # 获取实况天气示例
    print("=== 获取实况天气 ===")
    current_data = weather_api.get_weather(city_codes["北京"], extensions="base")
    if current_data:
        current_weather = weather_api.parse_current_weather(current_data)
        if current_weather:
            print(f"城市: {current_weather['city']}")
            print(f"天气: {current_weather['weather']}")
            print(f"温度: {current_weather['temperature']}°C")
            print(f"风向: {current_weather['wind_direction']}")
            print(f"风力: {current_weather['wind_power']}级")
            print(f"湿度: {current_weather['humidity']}")
            print(f"更新时间: {current_weather['report_time']}")

    print("\n=== 获取预报天气 ===")
    # 获取预报天气示例
    forecast_data = weather_api.get_weather(city_codes["北京"], extensions="all")
    if forecast_data:
        forecast_weather = weather_api.parse_forecast_weather(forecast_data)
        if forecast_weather:
            print(f"城市: {forecast_weather['city']}")
            print(f"预报时间: {forecast_weather['report_time']}")
            print("\n未来几天天气:")
            for forecast in forecast_weather['forecasts']:
                print(f"\n日期: {forecast['date']} (周{forecast['week']})")
                print(f"白天: {forecast['day_weather']}, 温度: {forecast['day_temp']}°C")
                print(f"夜间: {forecast['night_weather']}, 温度: {forecast['night_temp']}°C")
                print(f"风向: 白天{forecast['day_wind']}, 夜间{forecast['night_wind']}")
                print(f"风力: 白天{forecast['day_power']}级, 夜间{forecast['night_power']}级")


if __name__ == "__main__":
    main()