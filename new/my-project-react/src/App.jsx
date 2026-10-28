import Navbar from "./components/Navbar"
import HeroSection from "./components/HeroSection"
import Feature from "./components/Feature"
import Workflow from "./components/Workflow";
import Pricing from "./components/Pricing";
import Testimonial from "./components/Testimonial";
import Footer from "./components/Footer";
function App() {
  return (
    <>
    <Navbar/>
  <div className="max-w-7xl mx-auto pt-20 px-6">  
  <HeroSection/>
  <Feature/>
  <Workflow/>
  <Pricing/>
  <Testimonial/>
  <Footer/>
  </div>
  
  </>
  )
}

export default App
