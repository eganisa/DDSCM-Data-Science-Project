from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import itertools
from sklearn.feature_selection import f_regression
import pandas as pd
import dask
import dask.dataframe as dd
import plotly
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.figure_factory as ff 
import plotly.express as px
import plotly.graph_objects as go 
import json, csv, os, pickle, joblib
import math
from math import *


import uuid
secret_key = uuid.uuid4().hex
print(secret_key)


app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = '8da96fb9a89d456aaf47f825eda11429'

# Tentukan folder untuk menyimpan file yang diunggah untuk diproses lebih lanjut
UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')

# Konfigurasikan upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 1. Load data
dataco = pd.read_csv('dataset/DataCoSupplyChainDataset.csv', encoding = 'ISO-8859-1')
FeatureList = ['Type', 'Benefit per order', 'Sales per customer', 'Delivery Status', 'Late_delivery_risk',
               'Category Name', 'Customer City', 'Customer Country', 'Customer Id', 'Customer Segment',
               'Customer State', 'Customer Zipcode', 'Department Name', 'Latitude', 'Longitude', 'Market',
               'Order City', 'Order Country', 'Order Customer Id', 'order date (DateOrders)', 'Order Id',
               'Order Item Cardprod Id', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Id',
               'Order Item Product Price', 'Order Item Profit Ratio', 'Order Item Quantity', 'Sales', 'Order Item Total',
               'Order Profit Per Order', 'Order Region', 'Order State', 'Order Status', 'Order Zipcode', 'Product Card Id',
               'Product Category Id', 'Product Description', 'Product Image', 'Product Name', 'Product Price',
               'Product Status', 'shipping date (DateOrders)', 'Shipping Mode'] #list kolom yang akan dipakai

df1 = dataco[FeatureList]
print(df1)



# Load model 
with open('model\pipeline_rf_model.pkl', 'rb') as file:
    loaded_model_rf = pickle.load(file)
    
# Load model 
with open('model\pipeline_rf_regresor_model.pkl', 'rb') as file:
    loaded_model_rf_regressor = pickle.load(file)

# Load preprocessing
prepro_filename = 'model\label_encoder.pkl'
with open(prepro_filename, 'rb') as file:
    loaded_prepro = pickle.load(file)

print(type(loaded_model_rf))
print(type(loaded_model_rf_regressor))


# Route untuk halaman awal
@app.route('/')
# Function untuk merender halaman index.html
def home():
    return render_template('index.html')

# Route untuk halaman about
@app.route('/about')
# Function untuk merender halaman about.html
def about():
    return render_template('about.html')

# Route untuk halaman dashboard -> Predictive
@app.route('/predictive')
# Function untuk merender halaman predictive.html
def predictive():
    global new_results_dict
    df_table = df1[0:5]
    # Mengkonversi pandas dataframe menjadi dictionary
    new_results_dict = df_table.to_dict('records')  
    return render_template('predictive.html', 
                           data=new_results_dict)

# Route untuk halaman dashboard -> Analytics
@app.route('/dashboard')
# Function untuk merender halaman dasshboard.html
def dashboard():
    # Variabel global untuk menyimpan hasil data, gambar, dan grafik dalam kondisi global
    global fig, fig1,fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10
    global total_customer, total_order, total_amount, total_profit

    # Menghitung total customer, total order, total amount, dan total profit
    total_customer = df1['Customer Id'].nunique()
    total_order = df1['Order Id'].nunique()
    total_amount = df1['Sales'].sum()
    total_amount = f'${total_amount:.2f}'
    total_profit = df1['Order Profit Per Order'].sum()
    total_profit = f'${total_profit:.2f}'


    ###===================== 20 Pelanggan Utama Berdasarkan Banyaknya Pesanan ==========================###
    # Menyimpan isi Customer ID ke Customer_ID_STR dengan tipe data string
    df1['Customer_ID_STR'] = df1['Customer Id'].astype(str)

    # Mengelompokkan data berdasarkan 'Customer_ID_STR dan Order Id', menghitung jumlah pesanan, dan mengurutkan dari yang terbanyak
    data_customers = df1.groupby(['Customer_ID_STR'])['Order Id'].count().reset_index(name='Number of Orders')
    data_customers = data_customers.sort_values(by='Number of Orders', ascending=False)

    custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33',
                    '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57']
    
    # Membuat plot dengan grafik Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=data_customers.head(20)['Customer_ID_STR'],
        x=data_customers.head(20)['Number of Orders'],
        orientation='h',
        marker_color=custom_colors[:len(data_customers)]
    ))

    # Atur layout dan tampilan
    fig.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                      width=500,
                      height=300, 
                      font=dict(size=14) 
                      )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig.update_xaxes(title_text='Customers')
    fig.update_yaxes(title_text='Number of Orders')
    fig.update_xaxes(color='black', showgrid=False)
    fig.update_yaxes(color='black', showgrid=False)
    fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    ###===================== 20 Pelanggan dengan Keuntungan Tertinggi dari Seluruh Pesanan ==========================###
    # Menyimpan isi Customer ID ke Customer_ID_STR dengan tipe data string
    df1['Customer_ID_STR'] = df1['Customer Id'].astype(str)

    # Mengelompokkan data berdasarkan 'Customer_ID_STR', menghitung keuntungan dan mengurutkan dari yang terbesar
    data_customers_profit = df1.groupby(['Customer_ID_STR'])['Order Profit Per Order'].sum().reset_index(name = 'Profit of Orders')
    data_customers_profit = data_customers_profit.sort_values(by = 'Profit of Orders', ascending = False)

    custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33',
                    '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57']
    
    # Membuat plot dengan grafik Plotly
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=data_customers_profit.head(20)['Customer_ID_STR'],
        x=data_customers_profit.head(20)['Profit of Orders'],
        orientation='h',
        marker_color=custom_colors[:len(data_customers_profit)]
    ))

    # Atur layout dan tampilan
    fig1.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=500,
                       height=300, 
                       font=dict(size=14) 
                       )
    fig1.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig1.update_xaxes(title_text='Customers')
    fig1.update_yaxes(title_text='Profit of Orders')
    fig1.update_xaxes(color='black', showgrid=False)
    fig1.update_yaxes(color='black', showgrid=False)
    fig1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)


    ###===================== Segmen Pelanggan ==========================###
    # Mengelompokkan data berdasarkan 'Customer segment', menghitung keuntungan dan mengurutkan dari yang terbanyak
    data_Customer_Segment = df1.groupby(['Customer Segment'])['Order Id'].count().reset_index(name = 'Number of Orders')
    data_Customer_Segment = data_Customer_Segment.sort_values(by = 'Number of Orders', ascending = False)

    # Membuat plot
    fig2 = px.pie(data_Customer_Segment,
                  values = 'Number of Orders',
                  names = 'Customer Segment',
                  width = 400, height = 300 , color_discrete_sequence = px.colors.sequential.RdBu
                 )
    
    # Atur layout dan tampilan
    fig2.update_layout(legend_font_color='black')
    fig2.update_traces(textfont=dict(color='#fff'))
    fig2.update_layout(legend_title_font=dict(color='black'))
    fig2.update_layout(title_font=dict(color='black'))
    fig2.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'
    })
    fig2.update_xaxes(color='black', showgrid=False) 
    fig2.update_yaxes(color='black', showgrid=False) 
    fig2.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, b=30, t=0), 
        legend=dict(orientation="h", yanchor="top", y=1.5, xanchor="center", x=0.5), 
        font=dict(size=14) 
    )
    fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    ###===================== Kategori ==========================###
    # Mengelompokkan data berdasarkan 'Category Name', menghitung keuntungan dan mengurutkan dari yang terbanyak
    data_Category_Name = df1.groupby(['Category Name'])['Order Id'].count().reset_index(name = 'Number of Orders')
    data_Category_Name = data_Category_Name.sort_values(by = 'Number of Orders', ascending = True)

    custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33',
                    '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57']
    
    # Membuat plot dengan grafik Plotly
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        y=data_Category_Name.head(20)['Category Name'],
        x=data_Category_Name.head(20)['Number of Orders'],
        orientation='h',
        marker_color=custom_colors[:len(data_Category_Name)]
    ))

    # Atur layout dan tampilan
    fig3.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=500,
                       height=300, 
                       font=dict(size=14) 
                      )
    fig3.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig3.update_xaxes(title_text='Number of Orders')
    fig3.update_yaxes(title_text='Categories')
    fig3.update_xaxes(color='black', showgrid=False)
    fig3.update_yaxes(color='black', showgrid=False)
    fig3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)


    ###===================== Fitur Geografis ==========================###
    # Mengelompokkan data berdasarkan 'order country dan order city', menghitung keuntungan dan mengurutkan menurun berdasarkan 'Profit of Orders'.
    df_geo = df1.groupby(['Order Country', 'Order City'])['Order Profit Per Order'].sum().reset_index(name = 'Profit of Orders')
    df_geo = df_geo.sort_values(by = 'Profit of Orders', ascending = False)

    # Membuatkan peta choropleth menggambarkan keuntungan dari pesanan di berbagai wilayah
    fig4 = px.choropleth(df_geo, locationmode = 'country names', locations = 'Order Country',
                        color = 'Profit of Orders', hover_name = 'Order Country',
                        color_continuous_scale = px.colors.sequential.Plasma
                        )
    # Atur layout dan tampilan
    fig4.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
                       width=800,
                       height=400, 
                       font=dict(size=14) 
                      )
    fig4.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig4.update_xaxes(color='black', showgrid=False)
    fig4.update_yaxes(color='black', showgrid=False)
    fig4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)


    ###===================== Analisis Penjualan ==========================###
    # Mengelompokkan data berdasarkan 'order country', menghitung total penjualan ('Sales') dan mengurutkan menurun berdasarkan 'Sales of Orders'.
    df_sales_country = df1.groupby(['Order Country'])['Sales'].sum().reset_index(name='Sales of Orders')
    df_sales_country = df_sales_country.sort_values(by='Sales of Orders', ascending=False)

    custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33']
    
    # Membuat plot dengan grafik Plotly
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        y=df_sales_country.head(10)['Order Country'],
        x=df_sales_country.head(10)['Sales of Orders'],
        orientation='h',
        marker_color=custom_colors[:len(df_sales_country)]
    ))

    # Atur layout dan tampilan
    fig5.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=500,
                       height=300, 
                       font=dict(size=14) 
                      )
    fig5.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig5.update_xaxes(title_text='Order Country')
    fig5.update_yaxes(title_text='Sales of Orders')
    fig5.update_xaxes(color='black', showgrid=False)
    fig5.update_yaxes(color='black', showgrid=False)
    fig5 = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)


    # Mengelompokkan data berdasarkan 'Product Name', menghitung total penjualan ('Sales') dan mengurutkan menurun berdasarkan 'Sales of Orders'.
    df_sales_product = df1.groupby(['Product Name'])['Sales'].sum().reset_index(name='Sales of Orders')
    df_sales_product = df_sales_product.sort_values(by= 'Sales of Orders', ascending= False)

    custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33']
    
    # Membuat plot dengan grafik Plotly
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        y=df_sales_product.head(10)['Product Name'],
        x=df_sales_product.head(10)['Sales of Orders'],
        orientation='h',
        marker_color=custom_colors[:len(df_sales_product)]
    ))

    # Atur layout dan tampilan
    fig6.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=1000,
                       height=300, 
                       font=dict(size=14) 
                      )
    fig6.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig6.update_xaxes(title_text='Sales of Orders')
    fig6.update_yaxes(title_text='Product Name')
    fig6.update_xaxes(color='black', showgrid=False)
    fig6.update_yaxes(color='black', showgrid=False)
    fig6 = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    # Mengelompokkan berdasarkan Nama Produk dan Status Pengiriman, menghitung total penjualan ('Sales') untuk setiap kombinasi produk dan status pengiriman.
    df_status_delivery = df1.groupby([ 'Product Name', 'Delivery Status'])['Sales'].sum().reset_index(name = 'Sales of Orders')
    df_status_delivery = df_status_delivery.sort_values(by= 'Sales of Orders', ascending= False)

    custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33']
    
    # Membuat plot dengan grafik Plotly
    fig7 = px.bar(df_status_delivery.head(10),
                  x = 'Sales of Orders',
                  y = 'Product Name',
                  color = 'Delivery Status')

    # Atur layout dan tampilan
    fig7.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=1000,
                       height=400, 
                       font=dict(size=14),
                       legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5)
                      )
    fig7.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig7.update_xaxes(title_text='Sales of Orders')
    fig7.update_yaxes(title_text='Product Name ')
    fig7.update_xaxes(color='black', showgrid=False)
    fig7.update_yaxes(color='black', showgrid=False)
    fig7 = json.dumps(fig7, cls=plotly.utils.PlotlyJSONEncoder)


    # Mengelompokkan berdasarkan Kategori Produk dan menghitung total penjualan ('Sales') untuk setiap kategori.
    df_sales_pr = df1.groupby(['Category Name'])['Sales'].sum().reset_index(name = 'Sales of Orders')
    df_sales_pr = df_sales_pr.sort_values(by = 'Sales of Orders', ascending = False)

    custom_colors = ['#FF5733', '#33FF57', '#5733FF', '#FFFF33', '#33FFFF', '#FF33FF', '#FF5733', '#33FF57', '#5733FF', '#FFFF33']
    
    # Membuat plot dengan grafik Plotly
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(
        y=df_sales_pr.head(10)['Category Name'],
        x=df_sales_pr.head(10)['Sales of Orders'],
        orientation='h',
        marker_color=custom_colors[:len(df_sales_pr)]
    ))

    # Atur layout dan tampilan
    fig8.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=500,
                       height=300, 
                       font=dict(size=14) 
                      )
    fig8.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig8.update_xaxes(title_text='Sales of Orders')
    fig8.update_yaxes(title_text='Category Names')
    fig8.update_xaxes(color='black', showgrid=False)
    fig8.update_yaxes(color='black', showgrid=False)
    fig8 = json.dumps(fig8, cls=plotly.utils.PlotlyJSONEncoder)

    # Mengelompokkan berdasarkan Tipe Produk dan nama produk juga menghitung total penjualan ('Sales') untuk setiap kategori.
    df_sales_tp = df1.groupby(['Type', 'Product Name'])['Sales'].sum().reset_index(name = 'Sales of Orders')
    df_sales_tp = df_sales_tp.sort_values(by = 'Sales of Orders', ascending = False)

    # Membuat plot dengan grafik Plotly
    fig9 = px.bar(df_sales_tp.head(10),
                  x = 'Sales of Orders',
                  y = 'Type',
                  color = 'Product Name')
    # Atur layout dan tampilan
    fig9.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=1000,
                       height=400, 
                       font=dict(size=14),
                      )
    fig9.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig9.update_xaxes(title_text='Sales of Orders')
    fig9.update_yaxes(title_text='Product Name')
    fig9.update_xaxes(color='black', showgrid=False)
    fig9.update_yaxes(color='black', showgrid=False)
    fig9 = json.dumps(fig9, cls=plotly.utils.PlotlyJSONEncoder)

    # Menganalisis data penjualan berdasarkan bulan dan kuartal, serta memvisualisasikan hasilnya dalam bentuk diagram batang
    # Melakukan pengolahan data tanggal pada DataFrame df yang berhubungan dengan kolom 'order date (DateOrders)'
    data_orderdate = df1[['order date (DateOrders)', 'Sales']]
    data_orderdate['order_date'] = pd.to_datetime(data_orderdate['order date (DateOrders)'])
    data_orderdate["Quarter"] = data_orderdate['order_date'].dt.quarter
    data_orderdate["Month"] = data_orderdate['order_date'].dt.month
    data_orderdate['MonthStr'] = data_orderdate['Month'].astype(str)
    data_orderdate['QuarterStr'] = data_orderdate['Quarter'].astype(str)
    df_sales_m = data_orderdate.groupby(['QuarterStr', 'MonthStr'])['Sales'].sum().reset_index(name='Sales of Orders')
    df_sales_m = df_sales_m.sort_values(by = 'Sales of Orders', ascending = False)

    # Membuat plot dengan grafik Plotly
    fig10 = px.bar(df_sales_m.head(10),
                   x = 'Sales of Orders',
                   y = 'QuarterStr',
                   color = 'MonthStr')
    # Atur layout dan tampilan
    fig10.update_layout(margin=dict(l=100, r=50, t=0, b=0), 
                       width=1000,
                       height=400, 
                       font=dict(size=14),
                      )
    fig10.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
    })
    fig10.update_xaxes(title_text='Sales of Orders')
    fig10.update_yaxes(title_text='QuarterStr')
    fig10.update_xaxes(color='black', showgrid=False)
    fig10.update_yaxes(color='black', showgrid=False)
    fig10 = json.dumps(fig10, cls=plotly.utils.PlotlyJSONEncoder)  

    # Return render template yang mana akan merender halaman testing
    # Mengirim data dari variabel data, data grafik, data gambar ke sisi client
    return render_template('dashboard.html', 
                           fig=fig,
                           fig1=fig1,
                           fig2=fig2,
                           fig3=fig3,
                           fig4=fig4,
                           fig5=fig5,
                           fig6=fig6,
                           fig7=fig7,
                           fig8=fig8,
                           fig9=fig9,
                           fig10=fig10,
                           total_profit=total_profit,
                           total_order=total_order,
                           total_amount=total_amount,
                           total_customer=total_customer)
                           

# Mengubah data kategori menjadi angka-angka menggunakan label encoding
le = LabelEncoder()            # Objek untuk melakukan encoding pada data kategori
def Labelencoder_feature(x):   # Deklarasi fungsi yang menerima satu argumen, yaitu x, yang merupakan data yang akan di-label encoding.
    le = LabelEncoder()
    x = le.fit_transform(x)    # Mengambil data x dan mengembalikan versi yang telah di-encode.
    return x                   # Nilai yang telah di-label encoding akan dikembalikan


# Route untuk halaman prediction file yang menjalankan method GET dan POST
# POST mengirim data yang diupload user ke bagian sistem (Backend) untuk diolah
# Data yang telah diolah oleh sistem yang kemudian dikirimkan kembali ke sisi client (Frontend) untuk ditampilkan
@app.route('/prediction', methods=['GET','POST'])
def prediction_file():
    # Variabel global untuk menyimpan hasil data, gambar, dan grafik dalam kondisi global
    global new_results_dict, fig1, fig2, fig3
    # Periksa apakah file sudah diunggah
    if request.method == 'POST':
        # Dapatkan file yang diunggah
        uploaded_df = request.files['file']
        # Dapatkan nama file dan simpan file ke UPLOAD_FOLDER
        data_filename = secure_filename(uploaded_df.filename)
        # Simpan jalur file yang diunggah di sesi
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        filepath = session['uploaded_data_file_path']

        # Muat data dari file CSV ke dalam variabel dict_data dengan format list
        dict_data = []
        with open(filepath, encoding="latin-1") as file:
            csv_file = csv.DictReader(file)
            for row in csv_file:
                dict_data.append(row)

        # Ubah daftar kamus menjadi Pandas DataFrame
        df_results = pd.DataFrame.from_dict(dict_data)
        df_results = df_results[0:20000]
        print(df_results)
        
        # Copy ke variabel dataframe baru
        df_modeling1 = df_results.copy()
        df_modeling2 = df_results.copy()
        
        # Mengambil spesifik kolom saja
        data = df_modeling1[['Type', 'Benefit per order', 'Sales per customer', 'Delivery Status', 'Late_delivery_risk', 'Category Name',
             'Customer City', 'Customer Country', 'Customer Segment', 'Customer State', 'Customer Zipcode',
             'Department Name', 'Market', 'Order City', 'Order Country', 'Customer Id',
             'order date (DateOrders)', 'Order Item Cardprod Id', 'Order Item Discount', 'Order Item Discount Rate',
             'Order Item Id', 'Order Item Profit Ratio', 'Sales', 'Order Status',
             'Order Item Total', 'Order Profit Per Order', 'Order Region', 'Order State', 'Order Zipcode',
             'Product Card Id', 'Product Category Id', 'Product Description', 'Product Image', 'Product Name', 'Product Price',
             'Product Status', 'shipping date (DateOrders)', 'Shipping Mode']]
        
        # Memisahkan data menjadi fitur (features) dan target untuk pelatihan dan pengujian model
        features1 = data.drop(columns=['Late_delivery_risk'])     # Data dari kolom 'Late_delivery_risk' dihapus dari DataFrame data dan sisanya dianggap sebagai fitur

        final_features1 = features1[['Type', 'Shipping Mode', 'Order Region','Customer City', 'shipping date (DateOrders)']]
        final_features1 = final_features1.apply(Labelencoder_feature)

        # Membuat prediksi menggunakan model yang dimuat
        predictions_delivery_status = loaded_model_rf.predict(final_features1)

        # Menambahkan prediksi ke DataFrame
        df_results['predictedDeliveryStatus'] = predictions_delivery_status

        # Mengambil subset kolom-kolom tertentu dari DataFrame df
        data_sales = df_modeling2[['Type', 'Benefit per order', 'Sales per customer',
                'Delivery Status', 'Late_delivery_risk', 'Category Name', 'Customer City', 'Customer Country',
                'Customer Id', 'Customer Segment',
                'Customer State', 'Customer Zipcode', 'Department Name', 'Latitude', 'Longitude',
                'Market', 'Order City', 'Order Country', 'Order Customer Id', 'order date (DateOrders)', 'Order Id',
                'Order Item Cardprod Id', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Id',
                'Order Item Product Price', 'Order Item Profit Ratio', 'Order Item Quantity', 'Sales', 'Order Item Total',
                'Order Profit Per Order', 'Order Region', 'Order State', 'Order Status', 'Order Zipcode', 'Product Card Id',
                'Product Category Id', 'Product Description', 'Product Image', 'Product Name', 'Product Price', 'Product Status',
                'shipping date (DateOrders)', 'Shipping Mode']]

        features2 = data_sales.drop(columns=['Sales', 'Order Item Quantity', 'Order Item Product Price'])

        # Membuat DataFrame baru dengan nama 'final_features' yang hanya berisi kolom-kolom yang terdaftar
        final_features2 = features2[['Order Id', 'Order Item Discount', 'Order Item Cardprod Id',
            'shipping date (DateOrders)', 'order date (DateOrders)',
            'Order Customer Id', 'Order Profit Per Order', 'Market',
            'Order Region', 'Order State', 'Order Item Total',
            'Department Name', 'Product Card Id', 'Customer Id',
            'Product Category Id', 'Product Image', 'Category Name',
            'Product Name', 'Product Price', 'Sales per customer',
            'Benefit per order', 'Order Zipcode', 'Order Item Id',
            'Order City', 'Customer Segment']]
        
        
        final_features2 = final_features2.apply(Labelencoder_feature)

        # Membuat prediksi menggunakan model yang dimuat
        predictions_sales = loaded_model_rf_regressor.predict(final_features2)

        # Menambahkan prediksi ke DataFrame
        df_results['predictedSales'] = predictions_sales

        print(df_results)

        # Mengkonversi pandas dataframe menjadi dictionary
        new_results_dict = df_results.to_dict('records')


        ###===================== Status Delivery Prediction ==========================###
        df_results['predictedDeliveryStatus'].replace({0: 'On Time', 1: 'Delay'}, inplace=True)
        count_of_classes = df_results['predictedDeliveryStatus'].value_counts().sort_index()
        total = float(len(df_results))
        # Hitung persentase
        percentage_of_classes = count_of_classes / total * 100

        # Plot bar chart
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=count_of_classes.index,
            y=count_of_classes.values,
            text=[f'{y} ({p:.2f}%)' for x, y, p in zip(count_of_classes.index, count_of_classes.values, percentage_of_classes)],
            textposition='outside',
            hoverinfo='text',
            marker=dict(color=['#DE3163', '#1f77b4'])
        ))

        # Atur layout dan tampilan
        fig1.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                           width=500,
                           height=400,
                           margin=dict(l=0, r=0, t=0, b=0))
        fig1.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
        })
        fig1.update_traces(hoverlabel_font_color='white')
        fig1.update_xaxes(color='black', showgrid=False)
        fig1.update_yaxes(color='black', showgrid=False)
        fig1.update_xaxes(title_text='Delivery Status')
        fig1.update_yaxes(title_text='Count')
        fig1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)


        ###===================== Sales Prediction ==========================###
        # Create a scatter plot
        fig2 = go.Figure()

        # Add actual and predicted lines
        fig2.add_trace(go.Scatter(x=df_results['Sales'], y=df_results['predictedSales'], mode='markers',  name='Predicted vs Actual'))
        fig2.add_trace(go.Scatter(x=df_results['Sales'] , y=df_results['Sales'], mode='markers', name='Actual'))

        # Customize layout
        fig2.update_layout(xaxis_title= 'Actual Score', yaxis_title = 'Predicted Score')

        # Atur layout dan tampilan
        fig2.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                           width=500,
                           height=400,
                           margin=dict(l=0, r=0, t=0, b=0))
        fig2.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
        })
        fig2.update_traces(hoverlabel_font_color='white')
        fig2.update_xaxes(color='black', showgrid=False)
        fig2.update_yaxes(color='black', showgrid=False)
        fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        ###===================== Status Delivery Vs Sales Prediction ==========================###
        df_results['Order Id'] = df_results['Order Id'].astype(int)
        df_results['predictedDeliveryStatus'].replace({'On Time':0, 'Delay':1}, inplace=True)
        fig3 = px.scatter(df_results, x="Order Id", y="predictedSales", color="Customer Segment", size="predictedDeliveryStatus",
                          hover_name="Product Name")
        fig3.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGray')),
                        selector=dict(mode='markers'))

        # Atur layout dan tampilan
        fig3.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
                           width=1200,
                           height=400,
                           margin=dict(l=0, r=0, t=0, b=0))
        fig3.update_layout({
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
        })
        fig3.update_traces(hoverlabel_font_color='white')
        fig3.update_xaxes(color='black', showgrid=False)
        fig3.update_yaxes(color='black', showgrid=False)
        fig3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # Return render template yang mana akan merender halaman testing
    # Mengirim data dari variabel data, data grafik, data gambar ke sisi client
    return render_template('result.html',
                           fig1=fig1,
                           fig2=fig2,
                           fig3=fig3,
                           data=new_results_dict)

if __name__ == "__main__":
    app.run(debug=True)