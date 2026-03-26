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

    # 3. VERİLERİ SAYIYA ÇEVİR (Barların çalışması için şart)
    # Tablodaki sayısal kolonları seçip sayı tipine zorluyoruz
    cols_to_fix = ['-5','-4','-3','-2','-1','RATING','RATING_ADET','OVERRIDE_ADET','+1','+2','+3','+4','+5']
    for col in cols_to_fix:
        if col in dfPsi.columns:
            dfPsi[col] = pd.to_numeric(dfPsi[col], errors='coerce').fillna(0)

    # 4. BARLAR İÇİN STİL LİSTESİ OLUŞTURMA
    conditional_styles = []
    
    for col in cols_to_fix:
        if col == 'RATING': continue # Rating kolonu bar olmasın
        
        max_val = dfPsi[col].max()
        if max_val == 0: max_val = 1
        
        # Renk Belirleme: Eksiler kırmızı, Artılar yeşil, Adetler mavi
        if '-' in col: color = "rgba(255, 0, 0, 0.2)" # Kırmızı
        elif '+' in col: color = "rgba(0, 128, 0, 0.2)" # Yeşil
        else: color = "rgba(0, 0, 255, 0.1)" # Mavi (Adetler için)

        # Her satır için bar yüksekliğini hesapla
        for i in range(len(dfPsi)):
            val = dfPsi.iloc[i][col]
            percent = (val / max_val) * 100
            
            conditional_styles.append({
                'if': {'filter_query': f'{{{col}}} = {val}', 'column_id': col},
                'background': f'linear-gradient(90deg, {color} {percent}%, transparent {percent}%)',
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