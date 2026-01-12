import oracledb
import os
import csv
import time

# 1. 配置 Instant Client 路径 (你下载的 DMG 内容)
# 请将下面的路径修改为你实际解压的路径
# 提示：可以在终端输入 pwd 查看当前路径
lib_dir = os.path.expanduser("D:\Software\Oracle\instantclient_23_0")

try:
    # 启用“厚模式” (Thick Mode)，加载你下载的驱动
    oracledb.init_oracle_client(lib_dir=lib_dir)
    print("✅ Instant Client 驱动加载成功")
except Exception as e:
    print(f"❌ 驱动加载失败: {e}")
    exit(1)

# 2. 数据库连接参数 (源自文档 Page 17)
db_config = {
    "user": "student2501212302",    # 替换为你的真实账号，例如 student210121... [cite: 403]
    "password": "student2501212302",        # 替换为你的密码
    "dsn": "219.223.208.52/orcl" # Host:Port/ServiceName [cite: 385, 388, 389]
}

# 输出文件名
file_info = "CSI1000_Basic_Info_Real.csv"       # 期货基本信息
file_fut  = "CSI1000_Futures_EOD_Real.csv"      # 期货行情
file_etf  = "H00852_SH_EOD.csv"        # ETF行情 (新增)

# ==========================================
# 2. 初始化与工具函数
# ==========================================
try:
    oracledb.init_oracle_client(lib_dir=lib_dir)
except Exception:
    pass

def fetch_and_save(cursor, sql, filename, description):
    print(f"\n🚀 开始任务: {description} ...")
    start_time = time.time()
    try:
        cursor.execute(sql)
        
        # 检查是否查询到列名
        if not cursor.description:
             print(f"⚠️  未获取到列信息 ({description})")
             return

        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        
        if not rows:
            print(f"⚠️  未查到数据 ({description}) - 可能表名不同或无权限")
            return

        print(f"📊 共找到 {len(rows)} 条记录。正在写入...")
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
            
        print(f"✅ 保存成功: {filename} (耗时 {time.time() - start_time:.2f}s)")
        
    except Exception as e:
        print(f"❌ 任务失败 ({description}): {e}")

# ==========================================
# 3. 核心逻辑
# ==========================================
try:
    connection = oracledb.connect(**db_config)
    print("✅ 数据库连接成功")

    with connection.cursor() as cursor:
        
        # -------------------------------------------------------
        # 任务一：获取期货基本信息 (剔除连续/仿真)
        # -------------------------------------------------------
        sql_basic = """
        SELECT * FROM FILESYNC.CFuturesDescription
        WHERE S_INFO_CODE LIKE 'IM____'       -- 限制为6位代码 (IM+YYMM)
          AND S_INFO_EXCHMARKET = 'CFFEX'     -- 中金所
          AND S_INFO_NAME NOT LIKE '%连续%'
        ORDER BY S_INFO_LISTDATE
        """
        fetch_and_save(cursor, sql_basic, file_info, "获取中证1000真实合约列表")

        # -------------------------------------------------------
        # 任务二：获取期货行情 (基于真实合约)
        # -------------------------------------------------------
        sql_fut_prices = """
        SELECT *
        FROM FILESYNC.CIndexFuturesEODPrices t1
        WHERE t1.S_INFO_WINDCODE IN (
            SELECT S_INFO_WINDCODE
            FROM FILESYNC.CFuturesDescription
            WHERE S_INFO_CODE LIKE 'IM____'    
              AND S_INFO_EXCHMARKET = 'CFFEX'
              AND S_INFO_NAME NOT LIKE '%连续%'
        )
        ORDER BY t1.S_INFO_WINDCODE, t1.TRADE_DT
        """
        fetch_and_save(cursor, sql_fut_prices, file_fut, "获取期货真实合约全历史行情")

        # -------------------------------------------------------
        # 任务三：获取ETF行情 (512100.SH)
        # 表名：CMFIndexEOD (封闭式基金日行情-ETF通常在此表)
        # -------------------------------------------------------
        sql_etf_prices = """
        SELECT *
        FROM FILESYNC.AINDEXEODPRICES
        WHERE S_INFO_WINDCODE = 'h00852.SH'
        ORDER BY TRADE_DT
        """
        fetch_and_save(cursor, sql_etf_prices, file_etf, "获取中证1000ETF(512100)历史行情")

    connection.close()
    print("\n✨ 所有数据下载完成！")

except oracledb.Error as e:
    print(f"❌ 发生全局错误: {e}")