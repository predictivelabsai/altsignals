"""
FastHTML Landing Page for AltSignals platform.
Clones the design and functionality of altindex.com.
"""

from fasthtml.common import *
import uvicorn

# Initialize FastHTML app
app, rt = fast_app(
    hdrs=(
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"),
        Style("""
            .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .card-hover { transition: transform 0.3s ease, box-shadow 0.3s ease; }
            .card-hover:hover { transform: translateY(-5px); box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
            .pulse-animation { animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
            .hero-pattern { background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E"); }
        """)
    )
)

@rt('/')
def get():
    return Html(
        Head(
            Title("Alternative Data for Investing - AltSignals"),
            Meta(name="description", content="Make better investment decisions with better data. Get unique AI stock picks, stock alerts, and thousands of alternative insights."),
            Meta(name="viewport", content="width=device-width, initial-scale=1.0")
        ),
        Body(
            # Navigation Header
            Nav(
                Div(
                    Div(
                        # Logo
                        A(
                            Div(
                                I(cls="fas fa-chart-line text-2xl text-blue-600"),
                                Span("AltSignals", cls="ml-2 text-xl font-bold text-gray-800"),
                                cls="flex items-center"
                            ),
                            href="/",
                            cls="flex items-center"
                        ),
                        
                        # Desktop Navigation
                        Div(
                            A("Datasets", href="#datasets", cls="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium"),
                            A("Top Stocks", href="#stocks", cls="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium"),
                            A("Stock Alerts", href="#alerts", cls="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium"),
                            A("AI Stock Picks", href="#ai-picks", cls="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium"),
                            cls="hidden md:flex space-x-4"
                        ),
                        
                        # Auth Buttons
                        Div(
                            A("Log in", href="/login", cls="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium"),
                            A("Sign up", href="/signup", cls="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700"),
                            cls="flex items-center space-x-2"
                        ),
                        
                        cls="flex items-center justify-between w-full"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="bg-white shadow-sm border-b"
            ),
            
            # Hero Section
            Section(
                Div(
                    Div(
                        Div(
                            H1("Make Better Investment Decisions With Better Data", 
                               cls="text-4xl md:text-6xl font-bold text-white mb-6 leading-tight"),
                            P("Get unique AI stock picks, stock alerts, and thousands of alternative insights. Complement your due diligence with AltSignals.",
                              cls="text-xl text-white mb-8 opacity-90 max-w-2xl"),
                            
                            # Search Bar
                            Div(
                                Input(
                                    type="text",
                                    placeholder="Search Stocks & Companies",
                                    cls="w-full px-6 py-4 text-lg rounded-l-lg border-0 focus:ring-2 focus:ring-blue-300 focus:outline-none"
                                ),
                                Button(
                                    I(cls="fas fa-search"),
                                    cls="bg-green-500 text-white px-6 py-4 rounded-r-lg hover:bg-green-600 transition-colors"
                                ),
                                cls="flex max-w-md mb-6"
                            ),
                            
                            P("Or ", 
                              A("Sign up", href="/signup", cls="text-blue-200 underline hover:text-white"),
                              " (for free) to get started.",
                              cls="text-white mb-8"
                            ),
                            
                            # Popular Links
                            Div(
                                P("Popular Stocks: ", cls="text-white mb-2"),
                                Div(
                                    A("Tesla", href="#", cls="bg-blue-500 text-white px-3 py-1 rounded-full text-sm mr-2 hover:bg-blue-600"),
                                    A("Apple", href="#", cls="bg-green-500 text-white px-3 py-1 rounded-full text-sm mr-2 hover:bg-green-600"),
                                    A("Nvidia", href="#", cls="bg-purple-500 text-white px-3 py-1 rounded-full text-sm mr-2 hover:bg-purple-600"),
                                    cls="mb-4"
                                ),
                                P("Trending Stocks: ", cls="text-white mb-2"),
                                Div(
                                    A("Reddit Stocks", href="#", cls="bg-orange-500 text-white px-3 py-1 rounded-full text-sm mr-2 hover:bg-orange-600"),
                                    A("Top Gainers", href="#", cls="bg-red-500 text-white px-3 py-1 rounded-full text-sm mr-2 hover:bg-red-600"),
                                    A("Penny Stocks", href="#", cls="bg-yellow-500 text-white px-3 py-1 rounded-full text-sm mr-2 hover:bg-yellow-600"),
                                    cls="mb-4"
                                ),
                                P("Popular Crypto: ", cls="text-white mb-2"),
                                Div(
                                    A("Bitcoin", href="#", cls="bg-yellow-600 text-white px-3 py-1 rounded-full text-sm mr-2 hover:bg-yellow-700"),
                                    A("Ethereum", href="#", cls="bg-indigo-500 text-white px-3 py-1 rounded-full text-sm mr-2 hover:bg-indigo-600"),
                                )
                            ),
                            
                            cls="w-full md:w-1/2"
                        ),
                        
                        # Hero Image/Dashboard Preview
                        Div(
                            Div(
                                Img(
                                    src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 600'%3E%3Crect width='800' height='600' fill='%23f8fafc'/%3E%3Crect x='50' y='50' width='700' height='500' rx='10' fill='white' stroke='%23e2e8f0'/%3E%3Ctext x='400' y='100' text-anchor='middle' font-family='Arial' font-size='24' fill='%23334155'%3EAltSignals Dashboard%3C/text%3E%3Crect x='100' y='150' width='600' height='300' fill='%23f1f5f9'/%3E%3Cpath d='M150 350 L250 300 L350 250 L450 280 L550 200 L650 220' stroke='%2306b6d4' stroke-width='3' fill='none'/%3E%3Ccircle cx='150' cy='350' r='4' fill='%2306b6d4'/%3E%3Ccircle cx='250' cy='300' r='4' fill='%2306b6d4'/%3E%3Ccircle cx='350' cy='250' r='4' fill='%2306b6d4'/%3E%3Ccircle cx='450' cy='280' r='4' fill='%2306b6d4'/%3E%3Ccircle cx='550' cy='200' r='4' fill='%2306b6d4'/%3E%3Ccircle cx='650' cy='220' r='4' fill='%2306b6d4'/%3E%3C/svg%3E",
                                    alt="AltSignals Dashboard Preview",
                                    cls="w-full h-auto rounded-lg shadow-2xl"
                                ),
                                cls="relative"
                            ),
                            cls="w-full md:w-1/2 mt-8 md:mt-0"
                        ),
                        
                        cls="flex flex-col md:flex-row items-center justify-between"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20"
                ),
                cls="gradient-bg hero-pattern min-h-screen flex items-center"
            ),
            
            # Live Data Cards
            Section(
                Div(
                    Div(
                        # Sentiment Card
                        Div(
                            Div(
                                Span("Sentiment", cls="text-sm font-medium text-purple-600 bg-purple-100 px-2 py-1 rounded"),
                                H3("Lucid Motors (LCID)", cls="text-lg font-semibold text-gray-800 mt-2"),
                                Div(
                                    Span("51", cls="text-2xl font-bold text-gray-900"),
                                    Span("9.8%", cls="text-green-500 text-sm ml-2"),
                                    cls="flex items-center mt-1"
                                ),
                                cls="p-4"
                            ),
                            cls="bg-white rounded-lg shadow-md card-hover"
                        ),
                        
                        # Reddit Mentions Card
                        Div(
                            Div(
                                Span("Reddit Mentions", cls="text-sm font-medium text-orange-600 bg-orange-100 px-2 py-1 rounded"),
                                H3("SPDR S&P 500 ETF (SPY)", cls="text-lg font-semibold text-gray-800 mt-2"),
                                Div(
                                    Span("654", cls="text-2xl font-bold text-gray-900"),
                                    Span("142.5%", cls="text-green-500 text-sm ml-2"),
                                    cls="flex items-center mt-1"
                                ),
                                cls="p-4"
                            ),
                            cls="bg-white rounded-lg shadow-md card-hover"
                        ),
                        
                        # Job Posts Card
                        Div(
                            Div(
                                Span("Job Posts", cls="text-sm font-medium text-green-600 bg-green-100 px-2 py-1 rounded"),
                                H3("Mondelez (MDLZ)", cls="text-lg font-semibold text-gray-800 mt-2"),
                                Div(
                                    Span("1,599", cls="text-2xl font-bold text-gray-900"),
                                    Span("30.5%", cls="text-green-500 text-sm ml-2"),
                                    cls="flex items-center mt-1"
                                ),
                                cls="p-4"
                            ),
                            cls="bg-white rounded-lg shadow-md card-hover"
                        ),
                        
                        # Employee Outlook Card
                        Div(
                            Div(
                                Span("Employee Business Outlook", cls="text-sm font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded"),
                                H3("Cincinnati Financial (CINF)", cls="text-lg font-semibold text-gray-800 mt-2"),
                                Div(
                                    Span("68", cls="text-2xl font-bold text-gray-900"),
                                    Span("12.8%", cls="text-green-500 text-sm ml-2"),
                                    cls="flex items-center mt-1"
                                ),
                                cls="p-4"
                            ),
                            cls="bg-white rounded-lg shadow-md card-hover"
                        ),
                        
                        cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 -mt-16 relative z-10"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
                ),
                cls="pb-20"
            ),
            
            # Better Data Section
            Section(
                Div(
                    Div(
                        Div(
                            H2("Better Investments with Better Data", cls="text-3xl md:text-4xl font-bold text-gray-900 mb-6"),
                            P("At AltSignals, we go beyond traditional financial data in our investment analysis, integrating a variety of alternative data points to provide a more comprehensive view. Our unique methodology encompasses insights such as job postings, website traffic, customer satisfaction, app downloads, and social media trends, in addition to our rigorous financial and technical insights and analysis.",
                              cls="text-lg text-gray-600 mb-8 leading-relaxed"),
                            A("Sign up", href="/signup", cls="bg-blue-600 text-white px-8 py-3 rounded-lg text-lg font-medium hover:bg-blue-700 transition-colors"),
                            cls="w-full md:w-1/2"
                        ),
                        
                        # Dashboard Image
                        Div(
                            Img(
                                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 600 400'%3E%3Crect width='600' height='400' fill='%23f8fafc'/%3E%3Crect x='20' y='20' width='560' height='360' rx='8' fill='white' stroke='%23e2e8f0'/%3E%3Ctext x='300' y='50' text-anchor='middle' font-family='Arial' font-size='18' fill='%23334155'%3EReal-time Analytics%3C/text%3E%3Crect x='50' y='80' width='500' height='250' fill='%23f1f5f9'/%3E%3Cpath d='M80 250 L150 200 L220 180 L290 220 L360 160 L430 140 L500 120' stroke='%2310b981' stroke-width='3' fill='none'/%3E%3Ccircle cx='80' cy='250' r='3' fill='%2310b981'/%3E%3Ccircle cx='150' cy='200' r='3' fill='%2310b981'/%3E%3Ccircle cx='220' cy='180' r='3' fill='%2310b981'/%3E%3Ccircle cx='290' cy='220' r='3' fill='%2310b981'/%3E%3Ccircle cx='360' cy='160' r='3' fill='%2310b981'/%3E%3Ccircle cx='430' cy='140' r='3' fill='%2310b981'/%3E%3Ccircle cx='500' cy='120' r='3' fill='%2310b981'/%3E%3Ccircle cx='520' cy='350' r='25' fill='%2306b6d4'/%3E%3Ctext x='520' y='355' text-anchor='middle' font-family='Arial' font-size='12' fill='white'%3E76%3C/text%3E%3C/svg%3E",
                                alt="Analytics Dashboard",
                                cls="w-full h-auto rounded-lg shadow-xl"
                            ),
                            cls="w-full md:w-1/2"
                        ),
                        
                        cls="flex flex-col md:flex-row items-center justify-between gap-12"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20"
                ),
                cls="bg-gray-50"
            ),
            
            # Stats Section
            Section(
                Div(
                    Div(
                        Div(
                            H3("40k+", cls="text-4xl font-bold text-white mb-2"),
                            P("registered members", cls="text-blue-100"),
                            cls="text-center"
                        ),
                        Div(
                            H3("100+", cls="text-4xl font-bold text-white mb-2"),
                            P("unique daily stock alerts", cls="text-blue-100"),
                            cls="text-center"
                        ),
                        Div(
                            H3("High", cls="text-4xl font-bold text-white mb-2"),
                            P("win-rate on AI Stock picks", cls="text-blue-100"),
                            cls="text-center"
                        ),
                        Div(
                            H3("100k+", cls="text-4xl font-bold text-white mb-2"),
                            P("daily insights", cls="text-blue-100"),
                            cls="text-center"
                        ),
                        cls="grid grid-cols-1 md:grid-cols-4 gap-8"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20"
                ),
                cls="bg-blue-900"
            ),
            
            # AI Stock Picks Section
            Section(
                Div(
                    Div(
                        H2("Unique AI stock picks", cls="text-3xl md:text-4xl font-bold text-gray-900 mb-6"),
                        P("Unlock smarter investing with AltSignals' AI Stock Picks. Boasting a 80% win rate, our advanced algorithms sift through thousands of stocks daily, offering you top-notch buy or sell signals. We don't just rely on traditional financial data – we delve deeper. By harnessing the power of both traditional and alternative data – from social media trends, to employee sentiment, to user trends – we paint a fuller picture, enabling sharper, more informed investment decisions.",
                          cls="text-lg text-gray-600 mb-8 leading-relaxed max-w-4xl"),
                        P("With AltSignals, you're not just keeping pace; you're staying ahead of the curve, armed with the insights that matter.",
                          cls="text-lg text-gray-600 mb-8 leading-relaxed max-w-4xl"),
                        A("Sign up", href="/signup", cls="bg-blue-600 text-white px-8 py-3 rounded-lg text-lg font-medium hover:bg-blue-700 transition-colors"),
                        cls="text-center"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20"
                ),
                cls="bg-white"
            ),
            
            # Testimonial Section
            Section(
                Div(
                    Div(
                        H2("Don't take our word for it. Take theirs.", cls="text-3xl md:text-4xl font-bold text-gray-900 mb-12 text-center"),
                        
                        Div(
                            Div(
                                P("As an investor who has just started this year, I've tried various Stock websites, but Altindex is the best alternative data provider. It offers invaluable insights, which I haven't seen other websites offer, and even allows me to see lobbying costs. Also, I have heard the AI-backed stock picking service has consistently delivered impressive results, and the stock portfolio alerts are great, so I see my portfolio performance relative to the market.",
                                  cls="text-lg text-gray-600 mb-6 italic"),
                                Div(
                                    P("Kenneth Thakur", cls="font-semibold text-gray-900"),
                                    P("Retail investor", cls="text-gray-500"),
                                    cls="text-center"
                                ),
                                cls="bg-white p-8 rounded-lg shadow-lg max-w-4xl mx-auto"
                            )
                        ),
                        cls="text-center"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20"
                ),
                cls="bg-gray-50"
            ),
            
            # CTA Section
            Section(
                Div(
                    Div(
                        H2("Made it all the way down here?", cls="text-3xl md:text-4xl font-bold text-white mb-6 text-center"),
                        P("Then it's time to try AltSignals and to make more informed investments.", cls="text-xl text-blue-100 mb-4 text-center"),
                        P("Every day we find thousands of insights and we bring everything to you in an easy-to-use dashboard.", cls="text-lg text-blue-100 mb-8 text-center"),
                        
                        # Final Search Bar
                        Div(
                            Input(
                                type="text",
                                placeholder="Search for a Company",
                                cls="w-full px-6 py-4 text-lg rounded-l-lg border-0 focus:ring-2 focus:ring-blue-300 focus:outline-none"
                            ),
                            Button(
                                I(cls="fas fa-search"),
                                cls="bg-green-500 text-white px-6 py-4 rounded-r-lg hover:bg-green-600 transition-colors"
                            ),
                            cls="flex max-w-md mx-auto"
                        ),
                        
                        cls="text-center"
                    ),
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20"
                ),
                cls="gradient-bg"
            ),
            
            # Footer
            Footer(
                Div(
                    Div(
                        # About Section
                        Div(
                            H3("About Us", cls="text-lg font-semibold text-gray-900 mb-4"),
                            P("AltSignals revolutionizes investing with advanced alternative data analytics, smart insights, and stock alerts, presented in an easy-to-use dashboard powered by comprehensive company data from across the internet.",
                              cls="text-gray-600 text-sm leading-relaxed"),
                            cls="w-full md:w-1/3"
                        ),
                        
                        # Links Section
                        Div(
                            H3("Quick Links", cls="text-lg font-semibold text-gray-900 mb-4"),
                            Div(
                                A("Top Stocks", href="#", cls="text-gray-600 hover:text-blue-600 block mb-2"),
                                A("Our Data", href="#", cls="text-gray-600 hover:text-blue-600 block mb-2"),
                                A("AI Stock Picks", href="#", cls="text-gray-600 hover:text-blue-600 block mb-2"),
                                A("Stock Alerts", href="#", cls="text-gray-600 hover:text-blue-600 block mb-2"),
                            ),
                            cls="w-full md:w-1/3"
                        ),
                        
                        # Contact Section
                        Div(
                            H3("AltSignals", cls="text-lg font-semibold text-gray-900 mb-4"),
                            P("Making alternative data accessible for better investment decisions.",
                              cls="text-gray-600 text-sm mb-4"),
                            Div(
                                A(I(cls="fab fa-twitter text-xl"), href="#", cls="text-gray-400 hover:text-blue-500 mr-4"),
                                A(I(cls="fab fa-linkedin text-xl"), href="#", cls="text-gray-400 hover:text-blue-700 mr-4"),
                                A(I(cls="fab fa-github text-xl"), href="#", cls="text-gray-400 hover:text-gray-900"),
                                cls="flex"
                            ),
                            cls="w-full md:w-1/3"
                        ),
                        
                        cls="flex flex-col md:flex-row justify-between gap-8"
                    ),
                    
                    # Copyright
                    Div(
                        P("© 2025 AltSignals. All rights reserved.", cls="text-gray-500 text-sm text-center"),
                        P("Legal Disclaimer: The information provided by AltSignals is solely for informational purposes and not a substitute for professional financial advice.",
                          cls="text-gray-400 text-xs text-center mt-2"),
                        cls="border-t border-gray-200 pt-8 mt-8"
                    ),
                    
                    cls="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12"
                ),
                cls="bg-white"
            ),
            
            cls="min-h-screen"
        )
    )

@rt('/signup')
def get():
    return Div(
        H1("Sign Up for AltSignals", cls="text-3xl font-bold text-center mb-8"),
        P("Join thousands of investors making better decisions with alternative data.", cls="text-center text-gray-600 mb-8"),
        A("← Back to Home", href="/", cls="text-blue-600 hover:text-blue-800"),
        cls="max-w-md mx-auto mt-20 p-8"
    )

@rt('/login')
def get():
    return Div(
        H1("Login to AltSignals", cls="text-3xl font-bold text-center mb-8"),
        P("Welcome back! Access your dashboard and insights.", cls="text-center text-gray-600 mb-8"),
        A("← Back to Home", href="/", cls="text-blue-600 hover:text-blue-800"),
        cls="max-w-md mx-auto mt-20 p-8"
    )

if __name__ == "__main__":
    print("🚀 Starting AltSignals Landing Page...")
    print("📍 Visit: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

