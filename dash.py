from dash.exceptions import PreventUpdate

def update_date_dropdown(rat_rea_bol, model, pathname):
    # 1. HATA KORUMASI: Seçim yapılmadıysa kodu çalıştırma
    if model is None or rat_rea_bol is None:
        raise PreventUpdate

    # 2. VERİ ÇEKME
    musteri_or_grup = "musteri" if pathname == "/musteri" else "grup"
    dfPsi = getRatingData(musteri_or_grup)

    if dfPsi.empty:
        return "Veri bulunamadı."
    # ... (üst kısımdaki dfPsi çekme kısmı aynı) ...

    # 3. VERİLERİ SAYIYA ÇEVİR (TOTAL satırını bozmadan)
    cols_to_fix = ['-5','-4','-3','-2','-1','RATING','RATING_ADET','OVERRIDE_ADET','+1','+2','+3','+4','+5']
    
    for col in cols_to_fix:
        if col in dfPsi.columns:
            # Sadece "TOTAL" olmayan satırları sayıya çeviriyoruz
            # Diğer satırları olduğu gibi (string olarak) bırakıyoruz
            dfPsi[col] = dfPsi[col].apply(lambda x: pd.to_numeric(x, errors='coerce') if x != "TOTAL" else x)

    # 4. BARLAR İÇİN STİL LİSTESİ (TOTAL satırına bar eklememek için kontrol ekliyoruz)
    conditional_styles = []
    
    for col in cols_to_fix:
        if col == 'RATING': continue 
        
        # Sayısal değerlerin maksimumunu bul (TOTAL'i dahil etme)
        numeric_values = pd.to_numeric(dfPsi[col], errors='coerce').dropna()
        real_max = numeric_values.max() if not numeric_values.empty else 1
        max_val = real_max * 1.2
        
        # Renk ve yön belirleme (Aynı kalıyor)
        if '-' in col: 
            color, direction = "rgba(255, 69, 0, 0.5)", "270deg"
        elif '+' in col: 
            color, direction = "rgba(50, 205, 50, 0.5)", "90deg"
        else: 
            color, direction = "rgba(30, 144, 255, 0.4)", "90deg"

        for i in range(len(dfPsi)):
            val = dfPsi.iloc[i][col]
            
            # EĞER DEĞER "TOTAL" DEĞİLSE VE SAYISALSA BAR EKLE
            if val != "TOTAL" and pd.notnull(pd.to_numeric(val, errors='coerce')):
                num_val = float(val)
                percent = (num_val / max_val) * 100
                
                conditional_styles.append({
                    'if': {'filter_query': f'{{{col}}} = {val}', 'column_id': col},
                    'background': f'linear-gradient({direction}, {color} 0%, {color} {percent}%, transparent {percent}%, transparent 100%)',
                })
            
            # TOTAL SATIRINI KALINLAŞTIRMAK İÇİN (Opsiyonel)
            if val == "TOTAL" or dfPsi.iloc[i]['RATING'] == "TOTAL":
                conditional_styles.append({
                    'if': {'row_index': i},
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgba(240, 240, 240, 0.5)' # Hafif gri arka plan
                })

    # 5. TABLOYU DÖNDÜR
    return dash_table.DataTable(
        id="table-filtered-psi",
        columns=[{'name': i, 'id': i} for i in dfPsi.columns],
        data=dfPsi.to_dict('records'),
        style_data_conditional=conditional_styles, # Barları buraya bağladık
        style_data={'color': 'black'}, # Sayıları siyah yaptık
        style_header={'fontWeight': 'bold', 'color': 'black'},
        style_table={'overflowX': 'auto'},
        style_cell={'minWidth': '80px', 'textAlign': 'center'}
    )