
import { PayPalButtons } from "@paypal/react-paypal-js"
import { useEffect,useState } from "react"
import React from 'react'
import axios from "axios"

export const PaypalButt = ({description,price, handleApproval}) => {
    return (    
        <PayPalButtons  style={{ 
            layout: 'horizontal',
            color:'gold'}}
            createOrder={(data, actions) => {
                return actions.order.create({
                    purchase_units: [{description: description, amount: {value:price,currency_code: "AUD"}}],
                });
            }}
            onApprove={async (data, actions) => {
                const order = await actions.order.capture();
                console.log('order', order);
                handleApproval(actions.orderID);
            }} 
        />
    

    )
}   