from fasthtml.common import *

# Create app with inline CSS to ensure styling works
app, rt = fast_app()

@rt("/")
def get():
    return Html(
        Head(
            Title("Alternative Data for Investing - AltSignals"),
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Link(rel="stylesheet", href="https://cdn.tailwindcss.com"),
            Style("""
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
                
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                }
                
                body {
                    line-height: 1.6;
                    color: #374151;
                    background-color: #ffffff;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 0 20px;
                }
                
                .header {
                    background: white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    position: sticky;
                    top: 0;
                    z-index: 50;
                    border-bottom: 1px solid #e5e7eb;
                }
                
                .nav {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1rem 0;
                }
                
                .logo {
                    display: flex;
                    align-items: center;
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: #1f2937;
                    text-decoration: none;
                }
                
                .nav-links {
                    display: flex;
                    gap: 2rem;
                    align-items: center;
                }
                
                .nav-link {
                    color: #6b7280;
                    text-decoration: none;
                    font-weight: 500;
                    padding: 0.5rem 1rem;
                    border-radius: 0.375rem;
                    transition: color 0.2s;
                }
                
                .nav-link:hover {
                    color: #1f2937;
                }
                
                .btn-primary {
                    background: #f97316;
                    color: white;
                    padding: 0.75rem 1.5rem;
                    border-radius: 0.5rem;
                    text-decoration: none;
                    font-weight: 600;
                    transition: background-color 0.2s;
                    border: none;
                    cursor: pointer;
                }
                
                .btn-primary:hover {
                    background: #ea580c;
                }
                
                .btn-secondary {
                    background: #10b981;
                    color: white;
                    padding: 0.75rem 1.5rem;
                    border-radius: 0 0.5rem 0.5rem 0;
                    border: none;
                    cursor: pointer;
                    font-weight: 600;
                }
                
                .hero {
                    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                    padding: 5rem 0;
                }
                
                .hero-content {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 3rem;
                    align-items: center;
                }
                
                .hero-text h1 {
                    font-size: 3.5rem;
                    font-weight: 700;
                    color: #1f2937;
                    line-height: 1.1;
                    margin-bottom: 1.5rem;
                }
                
                .hero-text p {
                    font-size: 1.25rem;
                    color: #6b7280;
                    margin-bottom: 2rem;
                    line-height: 1.6;
                }
                
                .search-bar {
                    display: flex;
                    margin-bottom: 2rem;
                    max-width: 400px;
                }
                
                .search-input {
                    flex: 1;
                    padding: 0.75rem 1rem;
                    border: 2px solid #d1d5db;
                    border-radius: 0.5rem 0 0 0.5rem;
                    font-size: 1rem;
                    outline: none;
                }
                
                .search-input:focus {
                    border-color: #10b981;
                }
                
                .tag-group {
                    margin-bottom: 1rem;
                }
                
                .tag-label {
                    color: #6b7280;
                    font-weight: 500;
                    margin-right: 0.5rem;
                }
                
                .tag {
                    display: inline-block;
                    padding: 0.25rem 0.75rem;
                    margin: 0.25rem;
                    border-radius: 1rem;
                    text-decoration: none;
                    font-size: 0.875rem;
                    font-weight: 500;
                    transition: all 0.2s;
                }
                
                .tag-blue { background: #dbeafe; color: #1e40af; }
                .tag-green { background: #dcfce7; color: #166534; }
                .tag-purple { background: #f3e8ff; color: #7c3aed; }
                .tag-orange { background: #fed7aa; color: #c2410c; }
                .tag-pink { background: #fce7f3; color: #be185d; }
                .tag-yellow { background: #fef3c7; color: #d97706; }
                
                .tag:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }
                
                .dashboard-mockup {
                    background: white;
                    border-radius: 1rem;
                    padding: 2rem;
                    box-shadow: 0 25px 50px rgba(0,0,0,0.15);
                    border: 1px solid #e5e7eb;
                }
                
                .dashboard-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1.5rem;
                }
                
                .dashboard-title {
                    font-size: 1.25rem;
                    font-weight: 600;
                    color: #1f2937;
                }
                
                .status-dot {
                    width: 12px;
                    height: 12px;
                    background: #10b981;
                    border-radius: 50%;
                }
                
                .chart-area {
                    background: linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%);
                    border-radius: 0.5rem;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    height: 150px;
                    position: relative;
                    overflow: hidden;
                }
                
                .chart-line {
                    position: absolute;
                    top: 50%;
                    left: 10%;
                    right: 10%;
                    height: 3px;
                    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
                    border-radius: 2px;
                    transform: translateY(-50%);
                }
                
                .stats-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1rem;
                    text-align: center;
                }
                
                .stat-value {
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: #059669;
                }
                
                .stat-label {
                    font-size: 0.875rem;
                    color: #6b7280;
                }
                
                .cards-section {
                    padding: 4rem 0;
                    background: #f9fafb;
                }
                
                .cards-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1.5rem;
                }
                
                .card {
                    background: white;
                    border-radius: 0.75rem;
                    padding: 1.5rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid #e5e7eb;
                    transition: all 0.3s ease;
                }
                
                .card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                }
                
                .card-badge {
                    display: inline-block;
                    padding: 0.25rem 0.75rem;
                    border-radius: 1rem;
                    font-size: 0.75rem;
                    font-weight: 600;
                    margin-bottom: 0.75rem;
                }
                
                .badge-purple { background: #f3e8ff; color: #7c3aed; }
                .badge-orange { background: #fed7aa; color: #c2410c; }
                .badge-green { background: #dcfce7; color: #166534; }
                .badge-pink { background: #fce7f3; color: #be185d; }
                
                .card-title {
                    font-weight: 700;
                    color: #1f2937;
                    margin-bottom: 0.5rem;
                }
                
                .card-value {
                    font-size: 2rem;
                    font-weight: 700;
                    color: #1f2937;
                }
                
                .card-change {
                    color: #059669;
                    font-weight: 600;
                    margin-left: 0.5rem;
                }
                
                .section {
                    padding: 5rem 0;
                }
                
                .section-alt {
                    background: #f9fafb;
                }
                
                .section-dark {
                    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                    color: white;
                }
                
                .section-title {
                    font-size: 2.5rem;
                    font-weight: 700;
                    color: #1f2937;
                    margin-bottom: 1.5rem;
                }
                
                .section-text {
                    font-size: 1.125rem;
                    color: #6b7280;
                    line-height: 1.7;
                    margin-bottom: 2rem;
                }
                
                .two-column {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 3rem;
                    align-items: center;
                }
                
                .stats-grid-large {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 2rem;
                    text-align: center;
                }
                
                .stat-large {
                    font-size: 3rem;
                    font-weight: 700;
                    color: white;
                    margin-bottom: 0.5rem;
                }
                
                .stat-desc {
                    color: #94a3b8;
                }
                
                .testimonial {
                    background: white;
                    border-radius: 1rem;
                    padding: 2rem;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    border: 1px solid #e5e7eb;
                    max-width: 800px;
                    margin: 0 auto;
                }
                
                .testimonial-text {
                    font-size: 1.125rem;
                    color: #374151;
                    font-style: italic;
                    line-height: 1.7;
                    margin-bottom: 1.5rem;
                }
                
                .testimonial-author {
                    display: flex;
                    align-items: center;
                }
                
                .author-avatar {
                    width: 48px;
                    height: 48px;
                    background: #3b82f6;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: 700;
                    margin-right: 1rem;
                }
                
                .author-name {
                    font-weight: 700;
                    color: #1f2937;
                }
                
                .author-title {
                    color: #6b7280;
                }
                
                .footer {
                    background: #f9fafb;
                    padding: 3rem 0;
                    border-top: 1px solid #e5e7eb;
                    text-align: center;
                    color: #6b7280;
                }
                
                .text-center { text-align: center; }
                
                @media (max-width: 768px) {
                    .hero-content,
                    .two-column {
                        grid-template-columns: 1fr;
                        gap: 2rem;
                    }
                    
                    .hero-text h1 {
                        font-size: 2.5rem;
                    }
                    
                    .nav-links {
                        display: none;
                    }
                    
                    .cards-grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .stats-grid-large {
                        grid-template-columns: repeat(2, 1fr);
                    }
                }
            """)
        ),
        Body(
            # Header
            Header(
                cls="header",
                children=[
                    Div(
                        cls="container",
                        children=[
                            Nav(
                                cls="nav",
                                children=[
                                    A("⚡ AltSignals", href="/", cls="logo"),
                                    Div(
                                        cls="nav-links",
                                        children=[
                                            A("Datasets", href="#", cls="nav-link"),
                                            A("Top Stocks", href="#", cls="nav-link"),
                                            A("Stock Alerts", href="#", cls="nav-link"),
                                            A("AI Stock Picks", href="#", cls="nav-link"),
                                            A("Log in", href="#", cls="nav-link"),
                                            A("Sign up", href="#", cls="btn-primary"),
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Hero Section
            Section(
                cls="hero",
                children=[
                    Div(
                        cls="container",
                        children=[
                            Div(
                                cls="hero-content",
                                children=[
                                    Div(
                                        cls="hero-text",
                                        children=[
                                            H1("Make Better Investment Decisions With Better Data"),
                                            P("Get unique AI stock picks, stock alerts, and thousands of alternative insights. Complement your due diligence with AltSignals."),
                                            
                                            Div(
                                                cls="search-bar",
                                                children=[
                                                    Input(placeholder="Search Stocks & Companies", cls="search-input"),
                                                    Button("🔍", cls="btn-secondary"),
                                                ]
                                            ),
                                            
                                            P("Or ", A("Sign up", href="#", style="color: #dc2626; font-weight: 600;"), " (for free) to get started."),
                                            
                                            Div(
                                                cls="tag-group",
                                                children=[
                                                    Span("Popular Stocks:", cls="tag-label"),
                                                    A("Tesla", href="#", cls="tag tag-blue"),
                                                    A("Apple", href="#", cls="tag tag-green"),
                                                    A("Nvidia", href="#", cls="tag tag-purple"),
                                                ]
                                            ),
                                            
                                            Div(
                                                cls="tag-group",
                                                children=[
                                                    Span("Trending Stocks:", cls="tag-label"),
                                                    A("Reddit Stocks", href="#", cls="tag tag-orange"),
                                                    A("Top Gainers", href="#", cls="tag tag-pink"),
                                                    A("Penny Stocks", href="#", cls="tag tag-yellow"),
                                                ]
                                            ),
                                            
                                            Div(
                                                cls="tag-group",
                                                children=[
                                                    Span("Popular Crypto:", cls="tag-label"),
                                                    A("Bitcoin", href="#", cls="tag tag-yellow"),
                                                    A("Ethereum", href="#", cls="tag tag-blue"),
                                                ]
                                            ),
                                        ]
                                    ),
                                    
                                    Div(
                                        cls="dashboard-mockup",
                                        children=[
                                            Div(
                                                cls="dashboard-header",
                                                children=[
                                                    Span("AltSignals Dashboard", cls="dashboard-title"),
                                                    Div(cls="status-dot")
                                                ]
                                            ),
                                            Div(
                                                cls="chart-area",
                                                children=[
                                                    Div(cls="chart-line")
                                                ]
                                            ),
                                            Div(
                                                cls="stats-grid",
                                                children=[
                                                    Div(
                                                        children=[
                                                            Div("76%", cls="stat-value"),
                                                            Div("Win Rate", cls="stat-label")
                                                        ]
                                                    ),
                                                    Div(
                                                        children=[
                                                            Div("$2.4M", cls="stat-value"),
                                                            Div("Portfolio", cls="stat-label")
                                                        ]
                                                    ),
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Cards Section
            Section(
                cls="cards-section",
                children=[
                    Div(
                        cls="container",
                        children=[
                            Div(
                                cls="cards-grid",
                                children=[
                                    Div(
                                        cls="card",
                                        children=[
                                            Span("Sentiment", cls="card-badge badge-purple"),
                                            H3("Lucid Motors (LCID)", cls="card-title"),
                                            Span("51", cls="card-value"),
                                            Span("9.8%", cls="card-change"),
                                        ]
                                    ),
                                    Div(
                                        cls="card",
                                        children=[
                                            Span("Reddit Mentions", cls="card-badge badge-orange"),
                                            H3("SPDR S&P 500 ETF (SPY)", cls="card-title"),
                                            Span("654", cls="card-value"),
                                            Span("142.5%", cls="card-change"),
                                        ]
                                    ),
                                    Div(
                                        cls="card",
                                        children=[
                                            Span("Job Posts", cls="card-badge badge-green"),
                                            H3("Mondelez (MDLZ)", cls="card-title"),
                                            Span("1,599", cls="card-value"),
                                            Span("30.5%", cls="card-change"),
                                        ]
                                    ),
                                    Div(
                                        cls="card",
                                        children=[
                                            Span("Employee Outlook", cls="card-badge badge-pink"),
                                            H3("Cincinnati Financial (CINF)", cls="card-title"),
                                            Span("68", cls="card-value"),
                                            Span("12.8%", cls="card-change"),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Better Investments Section
            Section(
                cls="section",
                children=[
                    Div(
                        cls="container",
                        children=[
                            Div(
                                cls="two-column",
                                children=[
                                    Div(
                                        children=[
                                            H2("Better Investments with Better Data", cls="section-title"),
                                            P("At AltSignals, we go beyond traditional financial data in our investment analysis, integrating a variety of alternative data points to provide a more comprehensive view. Our unique methodology encompasses insights such as job postings, website traffic, customer satisfaction, app downloads, and social media trends.", cls="section-text"),
                                            A("Sign up", href="#", cls="btn-primary"),
                                        ]
                                    ),
                                    Div(
                                        style="background: #1f2937; border-radius: 1rem; padding: 2rem; box-shadow: 0 25px 50px rgba(0,0,0,0.15);",
                                        children=[
                                            Div(
                                                style="background: #374151; border-radius: 0.5rem; padding: 1.5rem; margin-bottom: 1rem; height: 120px; position: relative;",
                                                children=[
                                                    Div(style="position: absolute; top: 50%; left: 10%; right: 10%; height: 3px; background: linear-gradient(90deg, #10b981, #059669); border-radius: 2px; transform: translateY(-50%);")
                                                ]
                                            ),
                                            Div(
                                                style="display: flex; justify-content: center;",
                                                children=[
                                                    Div(
                                                        style="background: #06b6d4; border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.125rem;",
                                                        children=["76%"]
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Stats Section
            Section(
                cls="section section-dark",
                children=[
                    Div(
                        cls="container",
                        children=[
                            Div(
                                cls="stats-grid-large",
                                children=[
                                    Div(
                                        cls="text-center",
                                        children=[
                                            H3("40k+", cls="stat-large"),
                                            P("registered members", cls="stat-desc"),
                                        ]
                                    ),
                                    Div(
                                        cls="text-center",
                                        children=[
                                            H3("100+", cls="stat-large"),
                                            P("unique daily stock alerts", cls="stat-desc"),
                                        ]
                                    ),
                                    Div(
                                        cls="text-center",
                                        children=[
                                            H3("High", cls="stat-large"),
                                            P("win-rate on AI Stock picks", cls="stat-desc"),
                                        ]
                                    ),
                                    Div(
                                        cls="text-center",
                                        children=[
                                            H3("100k+", cls="stat-large"),
                                            P("daily insights", cls="stat-desc"),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # AI Stock Picks Section
            Section(
                cls="section",
                children=[
                    Div(
                        cls="container text-center",
                        children=[
                            H2("Unique AI stock picks", cls="section-title"),
                            P("Unlock smarter investing with AltSignals' AI Stock Picks. Boasting a 80% win rate, our advanced algorithms sift through thousands of stocks daily, offering you top-notch buy or sell signals.", cls="section-text"),
                            P("With AltSignals, you're not just keeping pace; you're staying ahead of the curve, armed with the insights that matter.", cls="section-text"),
                            A("Sign up", href="#", cls="btn-primary"),
                        ]
                    )
                ]
            ),
            
            # Testimonial Section
            Section(
                cls="section section-alt",
                children=[
                    Div(
                        cls="container",
                        children=[
                            H2("Don't take our word for it. Take theirs.", style="font-size: 2.5rem; font-weight: 700; color: #1f2937; margin-bottom: 3rem; text-align: center;"),
                            Div(
                                cls="testimonial",
                                children=[
                                    P("As an investor who has just started this year, I've tried various Stock websites, but AltSignals is the best alternative data provider. It offers invaluable insights, which I haven't seen other websites offer, and the AI-backed stock picking service has consistently delivered impressive results.", cls="testimonial-text"),
                                    Div(
                                        cls="testimonial-author",
                                        children=[
                                            Div("KT", cls="author-avatar"),
                                            Div(
                                                children=[
                                                    P("Kenneth Thakur", cls="author-name"),
                                                    P("Retail investor", cls="author-title"),
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Final CTA Section
            Section(
                cls="section text-center",
                children=[
                    Div(
                        cls="container",
                        children=[
                            H2("Made it all the way down here?", cls="section-title"),
                            P("Then it's time to try AltSignals and to make more informed investments.", style="font-size: 1.25rem; color: #6b7280; margin-bottom: 0.5rem;"),
                            P("Every day we find thousands of insights and we bring everything to you in an easy-to-use dashboard.", style="font-size: 1.125rem; color: #6b7280; margin-bottom: 2rem;"),
                        ]
                    )
                ]
            ),
            
            # Footer
            Footer(
                cls="footer",
                children=[
                    Div(
                        cls="container",
                        children=[
                            P("© 2025 AltSignals. All rights reserved.")
                        ]
                    )
                ]
            ),
        )
    )

if __name__ == '__main__':
    serve()

