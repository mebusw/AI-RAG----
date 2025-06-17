import streamlit as st
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString # Import NavigableString
import os
import re
import pandas as pd
from urllib.parse import urljoin, urlparse
import time
import hashlib # 用于生成更唯一的图片文件名

# --- 配置 ---
BASE_URL = 'https://www.ielts-mentor.com/writing-sample/writing-task-2'
OUTPUT_DIR = 'output'
MARKDOWN_DIR = os.path.join(OUTPUT_DIR, 'markdown')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

## TODO
help="""
ielts-mentor.com网站上的作文为3类：
- A类小作文。目录链接 https://www.ielts-mentor.com/writing-sample/academic-writing-task-1?start=0，详情链接形如 https://www.ielts-mentor.com/writing-sample/academic-writing-task-1/1001-numbers-of-male-and-female-research-students-at-us-university
- G类小作文，目录链接 https://www.ielts-mentor.com/writing-sample/gt-writing-task-1，详情链接形如 https://www.ielts-mentor.com/writing-sample/gt-writing-task-1/4117-you-started-in-your-present-job-two-years-ago
- 大作文，目录链接 https://www.ielts-mentor.com/writing-sample/writing-task-2，详情链接形如 https://www.ielts-mentor.com/writing-sample/writing-task-2/4083-success-in-life-comes-from-hard-work-dedication-and-motivation

保存下来的图文，需要处理，比如恢复图片链接，以及利用正则去除 'Rating' 字样之后无用评论
"""

# 请求头，模拟浏览器访问
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
}

# 确保目录存在
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- 辅助函数 ---
def slugify(text):
    """将文本转换为适合文件名的 slug"""
    text = str(text).strip().lower() # 确保是字符串
    text = re.sub(r'[^\w\s-]', '', text) # 移除特殊字符
    text = re.sub(r'[-\s]+', '-', text) # 替换空格和多个连字符为单个连字符
    return text

def download_image(image_url, file_prefix):
    """下载图片并返回本地路径，用于 Markdown 嵌入"""
    # 确保图片 URL 是绝对路径
    if not image_url.startswith(('http://', 'https://')):
        image_url = urljoin(BASE_URL, image_url)

    try:
        response = requests.get(image_url, stream=True, timeout=10, headers=HEADERS)
        response.raise_for_status() # 检查 HTTP 错误

        # 从 URL 解析文件名
        parsed_url = urlparse(image_url)
        path = parsed_url.path
        
        # 尝试从路径中获取文件名和扩展名
        filename_raw = os.path.basename(path)
        name, ext = os.path.splitext(filename_raw)

        # 如果没有有效的扩展名，或者文件名看起来不像文件（例如目录名），则生成一个哈希名
        if not ext or len(ext) > 5 or not name: # 简单判断是否是有效的扩展名
            # 使用 URL 的哈希值作为文件名的一部分，确保唯一性
            filename_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
            ext = '.png' # 默认一个常见的图片扩展名
            filename = f"{filename_hash}{ext}"
        else:
            filename = f"{name}{ext}"

        # 结合题目slug和图片原始文件名
        # 限制文件名的总长度，避免操作系统限制
        base_filename_prefix = slugify(file_prefix)[:50] # Use the file prefix (ID-title_slug)
        final_filename = f"{base_filename_prefix}_{filename}"
        
        image_path = os.path.join(IMAGES_DIR, final_filename)
        
        # 确保文件名是唯一的，避免覆盖已存在的同名文件
        counter = 1
        original_image_path = image_path
        while os.path.exists(image_path):
            name_part, ext_part = os.path.splitext(original_image_path)
            image_path = f"{name_part}_{counter}{ext_part}"
            counter += 1

        with open(image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # 返回相对于 MARKDOWN_DIR 的路径，以便在 Markdown 中正确引用
        return os.path.relpath(image_path, MARKDOWN_DIR) 
    except requests.exceptions.RequestException as e:
        st.warning(f"下载图片失败 ({image_url}): {e}")
        return None
    except Exception as e:
        st.warning(f"处理图片失败 ({image_url}): {e}")
        return None

def scrape_ielts_samples_to_markdown(progress_bar_placeholder, status_text_placeholder):
    all_links = set()
    st.markdown("---") # 分隔线
    st.subheader("🕸️ 爬虫调试信息")
    debug_info_container = st.empty() # 用于动态更新调试信息

    all_links_file_path = os.path.join(OUTPUT_DIR, "all_links.md")
    links_loaded_from_file = False

    if os.path.exists(all_links_file_path):
        status_text_placeholder.text(f"发现链接文件: {all_links_file_path}，尝试读取...")
        debug_info_container.info(f"尝试从 {all_links_file_path} 读取链接...")
        try:
            with open(all_links_file_path, 'r', encoding='utf-8') as f:
                loaded_links = [line.strip() for line in f if line.strip() and line.strip().startswith('http')]
            if loaded_links:
                all_links = set(loaded_links)
                links_loaded_from_file = True
                st.sidebar.info(f"已从 {all_links_file_path} 加载 {len(all_links)} 个链接。")
                debug_info_container.success(f"成功从 {all_links_file_path} 加载 {len(all_links)} 个链接。")
            else:
                st.sidebar.warning(f"{all_links_file_path} 为空或不包含有效链接。将尝试在线抓取。")
                debug_info_container.warning(f"{all_links_file_path} 为空或不包含有效链接。将尝试在线抓取。")
        except Exception as e:
            st.sidebar.error(f"读取 {all_links_file_path} 失败: {e}。将尝试在线抓取。")
            debug_info_container.error(f"读取 {all_links_file_path} 失败: {e}。将尝试在线抓取。")

    if not links_loaded_from_file:
        status_text_placeholder.text("正在在线抓取题目链接...")
        current_page = 0
        all_links = set() #确保从头开始收集

        while True:
            url = f"{BASE_URL}?start={current_page * 20}"
            debug_info_container.info(f"正在尝试访问分页 URL: `{url}`")
            status_text_placeholder.text(f"正在访问分页: {url}")
            
            page_links_found_count = 0
            new_links_added_count = 0

            try:
                response = requests.get(url, timeout=10, headers=HEADERS)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                debug_info_container.code(f"页面内容片段 (URL: {url}):\n{response.text[:500]}...", language='html')

                links_on_page_elements = soup.select('form#adminForm table.category td.list-title a')
                page_links_found_count = len(links_on_page_elements)
                debug_info_container.info(f"在 `{url}` 页面找到 `{page_links_found_count}` 个 `form#adminForm table.category td.list-title a` 元素。")
                
                if not links_on_page_elements and current_page > 0:
                    st.info(f"在 `{url}` 未找到任何链接元素，可能已到达最后一页。停止分页遍历。")
                    break

                found_valid_links_on_this_page = []
                for link_tag in links_on_page_elements:
                    href = link_tag.get('href')
                    if href and '/writing-sample/writing-task-2/' in href and not href.endswith('/writing-task-2'):
                        full_url = urljoin("https://www.ielts-mentor.com", href)
                        if full_url not in all_links:
                            all_links.add(full_url)
                            new_links_added_count += 1
                            found_valid_links_on_this_page.append(full_url)
                
                if found_valid_links_on_this_page:
                    debug_info_container.write(f"本页新增题目链接 ({new_links_added_count} 个):")
                    for link_item in found_valid_links_on_this_page: # Renamed to avoid conflict
                        debug_info_container.text(f"- {link_item}")
                else:
                    debug_info_container.info("本页未发现新的有效题目链接。")

                if new_links_added_count == 0 and page_links_found_count == 0 and current_page > 0:
                     st.info(f"在 `{url}` 未找到任何新的题目链接，且页面元素为空。停止分页遍历。")
                     break

                current_page += 1
                st.sidebar.markdown(f'<p class="text-sm text-gray-600">已发现 <strong>{len(all_links)}</strong> 个题目链接 (本页新增: {new_links_added_count})</p>', unsafe_allow_html=True)
                time.sleep(2)

            except requests.exceptions.RequestException as e:
                st.error(f"访问分页 `{url}` 时发生网络错误: {e}")
                debug_info_container.error(f"网络错误详情: {e}")
                break
            except Exception as e:
                st.error(f"处理分页 `{url}` 时发生未知错误: {e}")
                debug_info_container.error(f"未知错误详情: {e}")
                break
        
        # 保存新抓取到的链接
        if all_links:
            try:
                with open(all_links_file_path, 'w', encoding='utf-8') as f:
                    for link_item in sorted(list(all_links)):
                        f.write(link_item + '\n')
                st.sidebar.info(f"新抓取的 {len(all_links)} 个题目链接已保存到: {all_links_file_path}")
                debug_info_container.success(f"新抓取的 {len(all_links)} 个题目链接已保存到: {all_links_file_path}")
            except Exception as e:
                st.sidebar.error(f"保存 {all_links_file_path} 失败: {e}")
                debug_info_container.error(f"保存 {all_links_file_path} 失败: {e}")
        elif not links_loaded_from_file: # 如果在线抓取也没有链接
             st.sidebar.warning("在线抓取未发现任何链接。")
             debug_info_container.warning("在线抓取未发现任何链接。爬虫将不会处理任何详情页面。")

    if not all_links:
        status_text_placeholder.warning("没有可供抓取的链接。请检查链接文件或网络连接。")
        debug_info_container.warning("最终链接列表为空，无法进行内容抓取。")
        return

    status_text_placeholder.text(f"总共发现/加载 {len(all_links)} 个题目链接，开始抓取详细内容...")
    debug_info_container.info(f"最终确认的总题目链接数: {len(all_links)}")

    total_links = len(all_links)
    crawled_count = 0

    # 排序链接以确保每次运行的抓取顺序一致，并提高可调试性
    for i, link_item in enumerate(sorted(list(all_links))): # Renamed to avoid conflict
        crawled_count += 1
        progress_percentage = (crawled_count / total_links) * 100
        progress_bar_placeholder.progress(int(progress_percentage))
        status_text_placeholder.text(f"正在抓取 ({crawled_count}/{total_links}): {link_item}")
        
        debug_info_container.info(f"正在抓取详情页面: `{link_item}`")

        try:
            parsed_url = urlparse(link_item)
            # Use the last part of the URL path for the markdown filename
            url_path_basename = os.path.basename(parsed_url.path)

            # Fallback if basename is empty or just a slash (e.g. for root paths if they were ever encountered)
            if not url_path_basename or url_path_basename == '/':
                url_path_basename = hashlib.md5(link_item.encode()).hexdigest()[:16] # Use a hash of the URL
                debug_info_container.warning(f"URL path basename for {link_item} was empty/invalid. Using hash: {url_path_basename}")

            final_markdown_filename = os.path.join(MARKDOWN_DIR, f"{url_path_basename}.md")

            # Check if the markdown file already exists using the new naming convention
            if os.path.exists(final_markdown_filename):
                st.info(f"文件已存在，跳过: {final_markdown_filename}")
                debug_info_container.info(f"文件 `{final_markdown_filename}` 已存在，跳过。")
                continue

            response = requests.get(link_item, timeout=10, headers=HEADERS)
            response.raise_for_status()
            page_soup = BeautifulSoup(response.text, 'html.parser')

            # Get title for page metadata and for constructing the image file prefix
            title_tag = page_soup.select_one('h1.title')
            title = title_tag.get_text(strip=True) if title_tag else "No Title Found"
            title_slug_from_page = slugify(title)

            # For image file prefix: parse ID from url_path_basename (if pattern exists) and use title_slug.
            # This ID helps in making image names more specific and tied to content.
            id_match_for_image = re.match(r'(\d+)-', url_path_basename) # url_path_basename is the base for the markdown filename
            file_id_for_image_prefix = id_match_for_image.group(1) if id_match_for_image else slugify(link_item)[:8] # Fallback for image ID part

            # Try to find article.item-page first, then div.item-page as a fallback
            content_container = page_soup.find('article', class_='item-page')
            if not content_container:
                content_container = page_soup.find('div', class_='item-page')
            
            markdown_content_parts = []
            if content_container:
                extravote_marker_div = content_container.find('div', class_='size-1 extravote')
                if extravote_marker_div:
                    debug_info_container.info(f"Found 'extravote' marker div ('div.size-1.extravote'). Content will be extracted before this element.")
                else:
                    debug_info_container.warning(f"Could not find 'extravote' marker div ('div.size-1.extravote'). Will attempt to process all content within the item-page container.")

                # 1. Collect relevant child nodes before the extravote marker
                nodes_before_extravote = []
                for node in content_container.children:
                    if not isinstance(node, (Tag, NavigableString)): # Use imported Tag and NavigableString
                        continue
                    if extravote_marker_div and node is extravote_marker_div:
                        debug_info_container.info("Reached 'extravote' div while collecting nodes. Stopping collection.")
                        break
                    nodes_before_extravote.append(node)

                if not nodes_before_extravote:
                    st.warning(f"No content found before 'extravote' div (or 'extravote' div was at the very beginning) for {link_item}")
                    debug_info_container.warning(f"No nodes collected before 'extravote' for {link_item}. Markdown file might be empty or contain only title/link.")
                
                # 2. Create a temporary soup with copies of these nodes to process them (images, then markdown)
                temp_processing_soup_parent = BeautifulSoup('<div></div>', 'html.parser').div
                for node_to_copy in nodes_before_extravote:
                    temp_processing_soup_parent.append(node_to_copy.__copy__()) # Append copies to isolate modifications

                # 3. Process images globally within this temporary soup (which contains content before extravote)
                img_tags_in_temp_soup = temp_processing_soup_parent.find_all('img')
                if img_tags_in_temp_soup:
                    debug_info_container.info(f"Processing {len(img_tags_in_temp_soup)} image(s) from collected content for {link_item}.")
                for img in img_tags_in_temp_soup:
                    src = img.get('src')
                    if src and not src.startswith('data:'): # Ensure it's a downloadable link
                        alt_text = img.get('alt', 'Image') # Get alt text or use default
                        local_image_path = download_image(src, f"{file_id_for_image_prefix}-{title_slug_from_page}")
                        if local_image_path:
                            debug_info_container.info(f"Image `{src}` downloaded, embedding as `{local_image_path}`.")
                            # Replace img tag with a placeholder string that includes all necessary info
                            img.replace_with(f"@@IMAGE_PLACEHOLDER@@{alt_text}@@{local_image_path}@@{src}@@")
                        else:
                            debug_info_container.warning(f"Image `{src}` download failed, removing from content.")
                            img.extract() # Remove the tag if download fails
                    else: # data URI, empty src, or other non-downloadable src
                        debug_info_container.info(f"Invalid, Base64, or non-downloadable image (src: '{src if src else 'None'}') removed.")
                        img.extract() # Remove the tag
                
                # 4. Convert children of this temporary, image-processed soup to Markdown
                # Collect text and placeholder strings
                raw_content_parts = []
                for element in temp_processing_soup_parent.children: # Added missing loop
                    if isinstance(element, Tag): # Use the imported Tag class directly
                        if element.name == 'p':
                            # Get text, ensuring spaces between inline elements, then strip leading/trailing whitespace
                            p_text = element.get_text(separator=' ', strip=True)
                            if p_text: # Only add if paragraph has content
                                raw_content_parts.append(p_text + '\n\n') # Changed to raw_content_parts
                        elif element.name and element.name.startswith('h'):
                            level = int(element.name[1]) # h1, h2, etc.
                            raw_content_parts.append('#' * level + ' ' + element.get_text(strip=True) + '\n\n')
                        elif element.name == 'ul' or element.name == 'ol':
                            list_items_md = []
                            for li in element.find_all('li', recursive=False): # Only direct children li
                                list_items_md.append(f"- {li.get_text(strip=True)}")
                            if list_items_md: # Only add list if it has items
                                raw_content_parts.append('\n'.join(list_items_md) + '\n\n') # Changed to raw_content_parts
                        elif element.name == 'table':
                            table_markdown = []
                            headers = [th.get_text(strip=True) for th in element.find_all('th')]
                            if headers:
                                table_markdown.append('| ' + ' | '.join(headers) + ' |')
                                table_markdown.append('|' + '----|' * len(headers))
                            
                            for row_tag in element.find_all('tr', recursive=False): # Only direct children tr
                                cells = [td.get_text(strip=True) for td in row_tag.find_all('td')]
                                if cells: # Only add row if it has cells
                                    table_markdown.append('| ' + ' | '.join(cells) + ' |')
                            if table_markdown: # Only add table if it has some content
                                raw_content_parts.append('\n'.join(table_markdown) + '\n\n') # Changed to raw_content_parts
                        elif element.name == 'strong':
                            # This handles <strong> tags that are direct children.
                            # Text within <p><strong>text</strong></p> is handled by p.get_text().
                            strong_text = element.get_text(strip=True)
                            if strong_text: raw_content_parts.append(f"**{strong_text}**")
                        elif element.name == 'em':
                            em_text = element.get_text(strip=True)
                            if em_text: raw_content_parts.append(f"*{em_text}*")
                        elif element.name == 'a':
                            href = element.get('href')
                            text = element.get_text(strip=True)
                            if href: # For links, typically we want the text, and the URL is already captured or not needed for this conversion
                                raw_content_parts.append(text) # Keep only text, or format as [text](href) if desired
                            else: # If 'a' tag has no href but has text
                                if text: raw_content_parts.append(text)
                        # Removed obsolete/misplaced block for "![Image]" handling

                    elif isinstance(element, NavigableString): # Process direct text nodes, use imported NavigableString
                        stripped_text = element.strip()
                        if stripped_text:
                            raw_content_parts.append(stripped_text + '\n\n')
                    elif isinstance(element, str) and element.startswith("@@IMAGE_PLACEHOLDER@@"): # Handle the placeholder string
                         raw_content_parts.append(element)

            else:
                st.warning(f"未能找到内容区域 ('article.item-page' or 'div.item-page'): {link_item}")
                debug_info_container.warning(f"未能找到内容区域 `article.item-page` or `div.item-page`，跳过此链接: {link_item}")
                continue

            # Join raw parts and replace image placeholders
            raw_markdown_content = "".join(raw_content_parts)
            final_markdown_content = re.sub(r"@@IMAGE_PLACEHOLDER@@(.*?)@@(.*?)@@(.*?)@@", 
                                            r"!\1(\2)\n\nOriginal Image Source\n\n", raw_markdown_content)

            final_markdown = f"# {title}\n\n"
            final_markdown += f"原文链接: [{link_item}]({link_item})\n\n" # Corrected link format
            final_markdown += final_markdown_content.strip() # Use processed content

            with open(final_markdown_filename, 'w', encoding='utf-8') as f:
                f.write(final_markdown)
            
            st.success(f"已保存: {final_markdown_filename}")
            debug_info_container.success(f"已成功保存 `{final_markdown_filename}`。")

        except requests.exceptions.RequestException as e:
            st.error(f"抓取 `{link_item}` 时发生网络错误: {e}")
            debug_info_container.error(f"抓取详情页面 `{link_item}` 失败: 网络错误: {e}")
        except Exception as e: # Catch any other unexpected errors
            st.error(f"处理 `{link_item}` 时发生错误: {e}")
            debug_info_container.error(f"处理详情页面 `{link_item}` 失败: {e}")
        time.sleep(2) # 增加礼貌性延迟

    status_text_placeholder.success("爬虫完成！所有数据已保存到本地文件。")
    debug_info_container.success("爬虫执行完毕！")

def load_local_data():
    """从本地 Markdown 文件加载数据，用于显示在 Streamlit 中"""
    data = []
    for filename in os.listdir(MARKDOWN_DIR):
        if filename.endswith(".md"):
            filepath = os.path.join(MARKDOWN_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    title_match = re.search(r'# (.+?)\n', content)
                    link_match = re.search(r'\[原文链接]\((.+?)\)', content)
                    
                    title = title_match.group(1).strip() if title_match else filename.replace('.md', '').replace('-', ' ').title()
                    source_url = link_match.group(1).strip() if link_match else "N/A"
                    
                    image_paths_in_md = re.findall(r'!\[.*?]\((.+?)\)', content)
                    
                    data.append({
                        "filename": filename,
                        "title": title,
                        "source_url": source_url,
                        "content": content,
                        "image_paths": image_paths_in_md # 保存从 Markdown 中解析出的相对路径
                    })
            except Exception as e:
                st.warning(f"加载文件 {filename} 失败: {e}")
    return pd.DataFrame(data)

# --- Streamlit UI ---
st.set_page_config(
    page_title="IELTS Academic Writing Task 1 爬虫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 使用 Tailwind CSS CDN
st.markdown("""
<head>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp {
    background-color: #f8fafc; /* Tailwind gray-50 */
}
.stButton>button {
    background-color: #3b82f6; /* Tailwind blue-500 */
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    font-weight: 600;
    transition: background-color 0.2s;
}
.stButton>button:hover {
    background-color: #2563eb; /* Tailwind blue-600 */
}
.stProgress > div > div {
    background-color: #22c55e; /* Tailwind green-500 */
}
.stSuccess {
    background-color: #d1fae5; /* Tailwind green-100 */
    color: #065f46; /* Tailwind green-800 */
    border-radius: 0.5rem;
    padding: 1rem;
}
.stWarning {
    background-color: #fffbeb; /* Tailwind yellow-100 */
    color: #92400e; /* Tailwind yellow-800 */
    border-radius: 0.5rem;
    padding: 1rem;
}
.stError {
    background-color: #fee2e2; /* Tailwind red-100 */
    color: #991b1b; /* Tailwind red-800 */
    border-radius: 0.5rem;
    padding: 1rem;
}
.stDataFrame {
    border-radius: 0.5rem;
    overflow: hidden;
}
/* For Markdown rendered images */
.stMarkdown img {
    max-width: 100%;
    height: auto;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="text-4xl font-bold text-center text-gray-800 mb-6">IELTS Academic Writing Task 1 题目爬虫</h1>', unsafe_allow_html=True)

st.sidebar.header("控制面板")
if st.sidebar.button("启动爬虫", key="start_crawler", help="点击开始抓取雅思写作Task 1题目并保存到本地"):
    st.sidebar.markdown('<p class="text-lg font-semibold text-blue-600">爬虫状态:</p>', unsafe_allow_html=True)
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    scrape_ielts_samples_to_markdown(progress_bar, status_text)

st.markdown('<hr class="my-8">', unsafe_allow_html=True)

st.markdown('<h2 class="text-3xl font-semibold text-gray-700 mb-4">已抓取数据</h2>', unsafe_allow_html=True)

# 从本地加载数据
df = load_local_data()

if not df.empty:
    st.write(f"本地共有 {len(df)} 条数据。")

    # 数据过滤和搜索
    search_term = st.text_input("搜索题目 (标题或内容中包含):", "")
    if search_term:
        df_filtered = df[df['title'].str.contains(search_term, case=False, na=False) | 
                         df['content'].str.contains(search_term, case=False, na=False)]
    else:
        df_filtered = df

    if not df_filtered.empty:
        st.dataframe(df_filtered[['title', 'source_url', 'filename']].style.set_properties(**{'font-size': '1.0em'}))

        # 导出数据概览
        st.download_button(
            label="下载数据概览为 CSV",
            data=df_filtered[['title', 'source_url', 'filename']].to_csv(index=False).encode('utf-8'),
            file_name="ielts_writing_task1_overview.csv",
            mime="text/csv",
            help="下载本地已抓取题目的概览信息"
        )
        
        st.markdown('<h3 class="text-2xl font-semibold text-gray-700 mt-8 mb-4">详细内容预览</h3>', unsafe_allow_html=True)
        
        # 使用 selectbox 显示标题，方便查看详情
        selected_title = st.selectbox("选择一个题目查看详细内容:", df_filtered['title'].tolist())
        
        if selected_title:
            selected_sample = df_filtered[df_filtered['title'] == selected_title].iloc[0]
            st.markdown(f'<h4 class="text-xl font-bold text-gray-800 mt-4">{selected_sample["title"]}</h4>', unsafe_allow_html=True)
            st.markdown(f'<p class="text-gray-600 text-sm">来源: <a href="{selected_sample["source_url"]}" target="_blank" class="text-blue-500 hover:underline">{selected_sample["source_url"]}</a></p>', unsafe_allow_html=True)
            
            st.markdown('<h5 class="text-lg font-semibold text-gray-700 mt-4">内容:</h5>', unsafe_allow_html=True)
            st.markdown(selected_sample['content'], unsafe_allow_html=True)
            
            if selected_sample['image_paths']:
                st.markdown('<h5 class="text-lg font-semibold text-gray-700 mt-4">图片 (本地路径):</h5>', unsafe_allow_html=True)
                for img_path in selected_sample['image_paths']:
                    st.markdown(f'- `{img_path}`')
            else:
                st.info("此题目没有图片或图片未成功下载/嵌入。")

    else:
        st.info("没有找到符合搜索条件的题目。")
else:
    st.info("本地还没有雅思写作题目。请点击左侧按钮启动爬虫。")