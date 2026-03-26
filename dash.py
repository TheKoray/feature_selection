def update_date_dropdown(rat_rea_bol, model, pathname):
    # 1. Sayfayı belirle
    musteri_or_grup = "musteri" if pathname == "/musteri" else "grup"
    
    # 2. Veriyi çek
    dfPsi = getRatingData(musteri_or_grup)
    
    if dfPsi is None or dfPsi.empty:
        return html.Div("Veri bulunamadı veya yüklenemedi.", style={'color': 'red', 'padding': '20px'})

    # 3. Renk barları için kolon gruplarını tanımla
    neg_cols = ['-1', '-2', '-3', '-4', '-5']
    pos_cols = ['+1', '+2', '+3', '+4', '+5']
    
    # Maksimum değerleri hesapla (Bar uzunlukları için %100 noktası)
    max_val = dfPsi[neg_cols + pos_cols + ['RATING_ADET', 'OVERRIDE_ADET']].max().max()
    if max_val <= 0: max_val = 1

    # 4. Koşullu Stil Listesini oluştur
    conditional_styles = []

    for col in dfPsi.columns:
        if col in ['RATING', 'TOTAL']: continue # Bu kolonlara bar ekleme
        
        # Her satır için bar yüzdesini hesapla
        for i in range(len(dfPsi)):
            val = dfPsi.iloc[i][col]
            if not isinstance(val, (int, float)): continue
            
            percent = (val / max_val) * 100
            
            # Renk seçimi
            color = "rgba(255, 0, 0, 0.2)" if col in neg_cols else \
                    "rgba(0, 128, 0, 0.2)" if col in pos_cols else \
                    "rgba(0, 0, 255, 0.15)" # Adet kolonları için mavi
            
            # Yön seçimi (Negatifler sağdan sola, diğerleri soldan sağa)
            direction = "to left" if col in neg_cols else "to right"
            
            conditional_styles.append({
                'if': {'filter_query': f'{{{col}}} = {val}', 'column_id': col},
                'background': f'linear-gradient({direction}, {color} {percent}%, transparent {percent}%)',
            })

    # 5. Tabloyu Döndür
    return dash_table.DataTable(
        id="table-filtered-psi",
        columns=[{'name': i, 'id': i} for i in dfPsi.columns],
        data=dfPsi.to_dict('records'),
        style_data_conditional=conditional_styles,
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_cell={
            'color': 'black',
            'minWidth': '80px', 'width': '100px', 'maxWidth': '150px',
            'textAlign': 'center',
            'padding': '5px',
            'fontFamily': 'sans-serif'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'color': 'black',
            'border': '1px solid #dee2e6'
        },
        style_data={'border': '1px solid #eee'}
    )