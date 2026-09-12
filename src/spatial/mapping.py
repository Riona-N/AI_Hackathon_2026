import folium

def make_hotspot_map(df, output_html="hotspot_map.html"):
    center = [df["twd97lat"].mean(), df["twd97lon"].mean()]
    m = folium.Map(location=center, zoom_start=10, control_scale=True)

    for _, r in df.dropna(subset=["twd97lat","twd97lon"]).iterrows():
        risk = str(r.get("risk_level", "UNKNOWN"))
        popup = f'''
        <b>{r.get("siteengname","Unknown site")}</b><br>
        Risk: {risk}<br>
        Score: {r.get("risk_score","NA")}<br>
        Date: {r.get("sampledate","NA")}
        '''
        folium.CircleMarker(
            [r["twd97lat"], r["twd97lon"]],
            radius=9,
            popup=folium.Popup(popup, max_width=300),
            tooltip=f'{r.get("siteengname","site")} — {risk}',
            fill=True
        ).add_to(m)

    m.save(output_html)
    return m
