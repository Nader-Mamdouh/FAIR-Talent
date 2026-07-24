using FAIR.Application.DTOs.Identity;
using FluentValidation;

namespace FAIR.Application.Validations.Authentication
{
    public class LoginUserValidator : AbstractValidator<Login>
    {
        public LoginUserValidator()
        {
            RuleFor(x => x.Username).NotEmpty().WithMessage("UserName is Required.");
               
            RuleFor(x => x.Password).NotEmpty().WithMessage("Password is Required.");
        }
    }
}
